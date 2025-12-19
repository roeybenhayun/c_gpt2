
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

/*** 
 * How to run this code 
 * nvcc emb_calc.cu -o emb_calc
 * ./emb_calc
 * ***/

#define token_length (512)
#define vocab_size (50257)
#define ctx_len (1024) 
#define  d_model (768)

float wte[vocab_size][d_model] = {};
float embeddings[ctx_len][d_model] = {};
float embeddings_gpu[ctx_len][d_model] = {};
float wpe[ctx_len][d_model] = {};

int token[token_length];

float (*wte_d)[d_model];
float (*embeddings_d)[d_model];
float (*wpe_d)[d_model];
int *token_d;


void init_data(){
    int i,j;
    
    memset(embeddings,0,ctx_len*d_model*sizeof(float));

    // 
    for (i=0; i<vocab_size; i++){
        for (j=0; j<d_model; j++){
            wte[i][j] = (float)rand()/RAND_MAX;
        }
    }
    for (i=0; i<ctx_len; i++){
        for (j=0; j<d_model; j++){
            wpe[i][j] = (float)rand()/RAND_MAX;
        }
    }

    for (i=0;i<token_length;i++){
        token[i] = rand()%vocab_size;
    }
}

void init_gpu(){
    cudaError_t err;
    err = cudaMalloc((void**)&wte_d, vocab_size * sizeof (*wte_d));
    err = cudaMalloc((void**)&embeddings_d, ctx_len * sizeof (*embeddings_d));
    err = cudaMalloc((void**)&wpe_d, ctx_len * sizeof (*wpe_d));
    err = cudaMalloc((void**)&token_d, token_length * sizeof(int));

}

void copy_to_gpu(){
    cudaMemcpy(wte_d,  wte, vocab_size * sizeof(*wte_d),cudaMemcpyHostToDevice);
    cudaMemcpy(wpe_d,  wpe, ctx_len * sizeof(*wpe_d),cudaMemcpyHostToDevice);
    cudaMemcpy(embeddings_d,  embeddings, ctx_len * sizeof(*embeddings_d),cudaMemcpyHostToDevice);
    cudaMemcpy(token_d,  token, token_length * sizeof(int),cudaMemcpyHostToDevice);

}



void calc_emb_cpu(){
    for (int i=0; i<token_length;i++){
        for (int j=0; j<d_model;j++){       
            embeddings[i][j] = wte[token[i]][j] + wpe[i][j];
        }
    }
}

__global__ void embedding_kernel(float (*wte_d)[d_model],float(*wpe_d)[d_model],int*token_d,float (*embeddings_d)[d_model],int length){

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
// gpu kernel
void calc_emb_gpu(){
    // 1. Define Block Size (Threads per block)
    // 16x16 = 256 threads per block. This is a standard, safe size.
    dim3 threadsPerBlock;
    threadsPerBlock.x = 16; 
    threadsPerBlock.y = 16;
    threadsPerBlock.z = 1;
    // 2. Define Grid Size (Number of blocks)
    // We divide total size by block size and round up using ceiling division: (N + block - 1) / block
    dim3 numBlocks;
    numBlocks.x = (token_length + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (d_model + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;


    printf("Launching Kernel with Grid(%d, %d) and Block(%d, %d)\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    // 3. Launch Kernel
    embedding_kernel<<<numBlocks, threadsPerBlock>>>(wte_d, wpe_d, token_d, embeddings_d, token_length);
    
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
    cudaMemcpy(embeddings_gpu, embeddings_d, ctx_len * sizeof(*embeddings_d), cudaMemcpyDeviceToHost);

    int errors = 0;
    for(int i = 0; i < token_length; i++) {
        for(int j = 0; j < d_model; j++) {
            float diff = fabs(embeddings[i][j] - embeddings_gpu[i][j]);
            if(diff > 1e-4) { // Allow small floating point error
                if(errors < 5) printf("Mismatch at [%d][%d]: CPU %f vs GPU %f\n", i, j, embeddings[i][j], embeddings_gpu[i][j]);
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
    calc_emb_cpu();
  
    printf("Computing GPU...\n");
    calc_emb_gpu();
        
    printf("Verifying...\n");
    verify_results();

    // Free GPU memory
    cudaFree(wte_d);
    cudaFree(wpe_d);
    cudaFree(embeddings_d);
    cudaFree(token_d);
}
