#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

/*** 
 * How to run this code 
 * nvcc layer_norm_gpu.cu -o layer_norm_gpu
 * ./emb_calc
 * ***/


#define token_length (512)
#define vocab_size (50257)
#define ctx_len (1024) 
#define  d_model (768)
#define THREADS_PER_BLOCK 1024 

const float eps = 0.00001;


float ln1_gamma[d_model] = {};
float ln1_beta[d_model] = {};


float input[ctx_len][d_model] = {};
float output[ctx_len][d_model] = {};
float output_gpu[ctx_len][d_model] = {};

float *ln1_gamma_d;
float *ln1_beta_d;

float (*input_d)[d_model];
float (*output_d)[d_model];

void init_data(){
    int i,j;
    
    memset(output,0,ctx_len*d_model*sizeof(float));
    memset(output_gpu,0,ctx_len*d_model*sizeof(float));

    // 
    for (i=0; i<d_model; i++){        
            ln1_gamma[i] = (float)rand()/RAND_MAX;
       
    }
    for (i=0; i<d_model; i++){        
            ln1_beta[i] = (float)rand()/RAND_MAX;
       
    }

    for (i=0; i<token_length; i++){
        for (j=0; j<d_model; j++){
            input[i][j] = (float)rand()/RAND_MAX;
        }
    }
}



static float mean_(float *x, int len){
    float sum = 0.0;
    for (int i=0; i<len; i++){
        sum += *(x+i);
    }
    return (sum/(float)len);
}
static float variance_(float *x,int len, float mean){
    float sum = 0.0;
    for (int i=0; i<len; i++){
        sum += pow(((*(x+i)) - mean),2);
    }
    return (sum/(float)len);
}

static void layernorm_2d_cpu(float *a, int a_r, int a_c, 
                    float * ln_gamma, float * ln_beta,
                    float * out, float epsilon){   
    for (int i=0; i < a_r; i++){
        float *row = a + i * a_c;
        float mean = mean_(row,a_c);
        float var = variance_(row,a_c,mean);
        for (int j=0; j<a_c; j++){
            *(out + i*a_c + j) = *(ln_gamma+j) * ((*(a + i*a_c + j) - mean)/sqrt(var+epsilon)) + *(ln_beta+j);
        }
    }
}


//              &embeddings[0][0], n_token, d_model
//layernorm_2d(input,n_tokens,d_model,tbp->ln1_gamma,tbp->ln1_beta, &X_norm[0][0],eps);
//layer[l].ln1_gamma = &layer_norm1_gamma_d[l][0];
//layer[l].ln1_beta = &layer_norm1_beta_d[l][0];

void init_gpu(){
    cudaError_t err;
    err = cudaMalloc((void**)&ln1_gamma_d, d_model * sizeof (float));
    err = cudaMalloc((void**)&ln1_beta_d, d_model * sizeof (float));
    err = cudaMalloc((void**)&input_d, ctx_len * sizeof (*input_d));
    err = cudaMalloc((void**)&output_d, ctx_len * sizeof(*output_d));
    if(err != cudaSuccess) printf("Alloc Error: %s\n", cudaGetErrorString(err));

}
void copy_to_gpu(){
    cudaMemcpy(ln1_gamma_d,  ln1_gamma, d_model * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(ln1_beta_d,  ln1_beta, d_model * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(input_d,  input, ctx_len * sizeof (*input_d),cudaMemcpyHostToDevice);
    cudaMemcpy(output_d,  output, ctx_len * sizeof(*output),cudaMemcpyHostToDevice);

}


void calc_layernorm_cpu(){
    layernorm_2d_cpu(&input[0][0],token_length,d_model,ln1_gamma,ln1_beta, &output[0][0],eps);

}

__global__ void layer_norm_kernel(float (*wte_d)[d_model],float(*wpe_d)[d_model],int*token_d,float (*embeddings_d)[d_model],int length){

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;    

    // 2. Boundary Check
    if (row < length && col < d_model) {
        
        // 3. Read input (Note: token_d is on GPU memory)
        int token_id = token_d[row];
        
        // 4. Compute
        // Using '_d' names here reminds us we are touching GPU memory!
        embeddings_d[row][col] = wte_d[token_id][col] + wpe_d[row][col];
    }

}


__global__ void layernorm_kernel(float (*input)[d_model], 
                                 float *gamma, 
                                 float *beta, 
                                 float (*output)[d_model], 
                                 int n_tokens,
                                float eps) {
    
    // 1. Setup Shared Memory
    // Enough for 1024 threads. "volatile" prevents the compiler from optimizing away memory reads during reduction
    __shared__ float s_data[THREADS_PER_BLOCK];

    // Calculate Identity
    int tid = threadIdx.x;      // My column (0..1023)
    int row_idx = blockIdx.x;   // My token (0..511)

    // Safety: If we have more blocks than rows, exit
    if (row_idx >= n_tokens) return;

    // --- PHASE 1: Calculate MEAN ---

    // 2. Load Data from Global to Register & Shared
    // Padding logic: If tid >= 768, we load 0.0 so we don't mess up the sum
    float val = 0.0f;
    if (tid < d_model) {
        val = input[row_idx][tid]; // Read Global
    }
    s_data[tid] = val; // Write Shared

    __syncthreads(); // Wait for everyone to fill shared mem

    // 3. Tree Reduction (Summation)
    // We fold the array in half: 1024 -> 512 -> 256 ... -> 1
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // 4. Thread 0 Calculates Mean
    // Result is now in s_data[0]
    if (tid == 0) {
        s_data[0] = s_data[0] / (float)d_model; 
    }
    __syncthreads(); // Wait for Thread 0

    // 5. Broadcast Mean to Private Register
    // CRITICAL: We save this in a register because s_data will be overwritten in Phase 2!
    float mean = s_data[0];


    // --- PHASE 2: Calculate VARIANCE ---

    // 6. Calculate Squared Difference
    // Use the 'val' we kept in our register from the start
    float diff_sq = 0.0f;
    if (tid < d_model) {
        float diff = val - mean;
        diff_sq = diff * diff;
    }
    
    // 7. Overwrite Shared Memory
    s_data[tid] = diff_sq; 
    
    __syncthreads();

    // 8. Tree Reduction (Summation of Diff Squared)
    // Same loop structure as before
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // 9. Thread 0 Calculates Variance (and std dev)
    // We put the inverse std dev back in s_data[0] for efficiency
    if (tid == 0) {
        float variance = s_data[0] / (float)d_model;
        // Pre-calculate the division factor: 1 / sqrt(var + eps)
        s_data[0] = rsqrtf(variance + eps); 
    }
    __syncthreads();

    // 10. Broadcast Inverse Std Dev
    float inv_std_dev = s_data[0];


    // --- PHASE 3: Final Normalize & Store ---
    
    // 11. Final Math
    // Only real columns write back (ignore the padding threads)
    if (tid < d_model) {
        // (x - mean) / std_dev
        float normalized = (val - mean) * inv_std_dev;

        // Apply Learnable Parameters (Gamma * norm + Beta)
        float result = normalized * gamma[tid] + beta[tid];

        // Write to Global Output
        output[row_idx][tid] = result;
    }
}

void calc_layernorm_gpu(){

    dim3 threadsPerBlock;
    threadsPerBlock.x = 1024;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    
    // 2. Define Grid Size (Number of blocks)
    // We divide total size by block size and round up using ceiling division: (N + block - 1) / block
    dim3 numBlocks;
    numBlocks.x = token_length;
    numBlocks.y = 1;
    numBlocks.z = 1;

    printf("Launching Kernel with Grid(%d, %d) and Block(%d, %d)\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    layernorm_kernel<<<numBlocks, threadsPerBlock>>>(input_d,ln1_gamma_d,ln1_beta_d,output_d, token_length,1e-5f);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

}
void verify_results() {
    // Copy result back from GPU to CPU to compare
    cudaMemcpy(output_gpu, output_d, ctx_len * sizeof(*output_d), cudaMemcpyDeviceToHost);

    int errors = 0;
    for(int i = 0; i < token_length; i++) {
        for(int j = 0; j < d_model; j++) {
            float diff = fabs(output[i][j] - output_gpu[i][j]);
            if(diff > 1e-4) { // Allow small floating point error
                if(errors < 5) printf("Mismatch at [%d][%d]: CPU %f vs GPU %f\n", i, j, output[i][j], output_gpu[i][j]);
                errors++;
            }
        }
    }

    if(errors == 0) printf("SUCCESS: CPU and GPU results match!\n");
    else printf("FAILURE: Found %d mismatches.\n", errors);
}

int main(){
    printf("Initializing data\n");
    init_data();
    printf("Initializing GPU\n");
    init_gpu();

    printf("Copy to GPU...\n");
    copy_to_gpu();

    printf("Computing CPU...\n");
    clock_t t0 = clock();
    calc_layernorm_cpu();
    printf("CPU Time: %.2f ms\n", (double)(clock()-t0)/CLOCKS_PER_SEC * 1000.0);
  
    printf("Computing GPU...\n");
    t0 = clock();
    calc_layernorm_gpu();
    printf("GPU Time: %.2f ms\n", (double)(clock()-t0)/CLOCKS_PER_SEC * 1000.0);
        
    printf("Verifying...\n");
    verify_results();

    // Free GPU memory
    cudaFree(ln1_gamma_d);
    cudaFree(ln1_beta_d);
    cudaFree(input_d);
    cudaFree(output_d);
}

