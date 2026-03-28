#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

/*** 
 * How to run this code 
 * nvcc softmax.cu -o softmax
 * ./emb_calc
 * ***/


#define ctx_len (1024) 
#define  d_model (768)

//#define ctx_len (2) //rows  
//#define  d_model (4) // columns

const float eps = 0.00001;
float input[ctx_len][d_model] = {};
float output[ctx_len][d_model] = {};
float output_gpu[ctx_len][d_model] = {};

float (*input_d)[d_model];
float (*output_d)[d_model];


void init_data(){
    int i=0,j=0;
    for (;i<ctx_len;i++){
        for (;j<d_model;j++){        
            input[i][j] = (float)rand()/RAND_MAX;
        }
        j=0;
    }
    memset(output,0,ctx_len*d_model*sizeof(float));
    memset(output_gpu,0,ctx_len*d_model*sizeof(float));
}

__global__ void softmax_kernel(float (*in)[d_model], float (*out)[d_model], int rows, int cols, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < rows) {
        float *row_ptr = in[i];
        float *out_ptr = out[i];

        // 1. Find Max
        float row_max = -INFINITY;
        for (int j = 0; j < cols; j++) {
            if (row_ptr[j] > row_max) row_max = row_ptr[j];
        }

        // 2. Sum Exponentials
        float sum_exp = 0.0f;
        for (int j = 0; j < cols; j++) {
            float e = expf(row_ptr[j] - row_max);
            out_ptr[j] = e;
            sum_exp += e;
        }

        // 3. Normalize
        for (int j = 0; j < cols; j++) {
            out_ptr[j] /= sum_exp;
        }
    }
}

void calc_softmax_gpu(){
    // 1. Define Block Size (Threads per block)
    
    dim3 threadsPerBlock;
    threadsPerBlock.x = 256; 
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    // 2. Define Grid Size (Number of blocks)
    // We divide total size by block size and round up using ceiling division: (N + block - 1) / block
    dim3 numBlocks;
    numBlocks.x = ((ctx_len + threadsPerBlock.x - 1) / threadsPerBlock.x);
    numBlocks.y = 1;
    numBlocks.z = 1;


    printf("Launching Kernel with Grid(%d, %d) and Block(%d, %d)\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    // 3. Launch Kernel
    softmax_kernel<<<numBlocks, threadsPerBlock>>>(input_d,output_d,ctx_len,d_model,d_model);
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

}

static void softmax_2d(float *a, int a_r, int a_c,int stride, float * c_out){    
    for (int i=0; i<a_r; i++){
        // 1. Find the maximum value in the current row
        float row_max = -INFINITY;
        for (int j=0; j<a_c; j++){
            if (*(a + i*stride + j) > row_max) {
                row_max = *(a + i*stride + j);
            }
        }
        // 2. Calculate the exponentials with the maximum subtracted and the sum
        float sum_exp = 0.0;
        for (int j = 0; j < a_c; j++) {
            float shifted = *(a + i * stride + j) - row_max;
            float exp_val = expf(shifted);
            *(c_out + i * stride + j) = exp_val;
            sum_exp += exp_val;
        }
        if (sum_exp == 0.0f && i == 1) {
            printf("WARNING: softmax sum_exp == 0 at row i=%d — input may be all -inf\n", i);
        }
        // 3. Normalize
        for (int j = 0; j < a_c; j++) {
            *(c_out + i * stride + j) /= sum_exp;
        }                 
    }
}


void print_array(float * in, int in_r, int in_c){
    int i,j;
    //int j = 0;
    for (i = 0;i<in_r;i++){
        for (j =0;j<in_c;j++){            
            printf("[%d][%d]: %f\n",i,j,*(in + i*in_c + j));
        }
        j = 0;
        printf("\n");
    }
}
void init_gpu(){
    cudaError_t err;    
    err = cudaMalloc((void**)&input_d, d_model*ctx_len * sizeof (*input_d));
    if(err != cudaSuccess) printf("Alloc Error: %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void**)&output_d, d_model * ctx_len * sizeof(*output_d));
    if(err != cudaSuccess) printf("Alloc Error: %s\n", cudaGetErrorString(err));

}
void copy_to_gpu(){
    cudaMemcpy(input_d,  input, ctx_len * sizeof (*input_d),cudaMemcpyHostToDevice);
    cudaMemcpy(output_d,  output, ctx_len * sizeof(*output),cudaMemcpyHostToDevice);
}

void verify_results() {
    // Copy result back from GPU to CPU to compare
    cudaMemcpy(output_gpu, output_d, ctx_len * sizeof(*output_d), cudaMemcpyDeviceToHost);

    int errors = 0;
    for(int i = 0; i < ctx_len; i++) {
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
    softmax_2d(&input[0][0],ctx_len, d_model,d_model,&output[0][0]);
  
    print_array(&output[0][0],ctx_len,d_model);

    printf("Computing GPU...\n");
    calc_softmax_gpu();
        
    //printf("Verifying...\n");
    verify_results();

    // Free GPU memory

    cudaFree(input_d);
    cudaFree(output_d);
    
}
