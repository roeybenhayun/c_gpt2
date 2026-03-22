#include <stdio.h>

#include <cuda_runtime.h>

#include "../include/cuda_kernels.h"

__global__ void embedding_kernel(float (*wte_d)[d_model],
                                 float (*wpe_d)[d_model],
                                 int *token_d,
                                 float (*embeddings_d)[d_model],
                                 int length) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;    

    // 2. Boundary Check
    if (row < length && col < d_model) {
        
        // 3. Read input (Note: token_d is on GPU memory)
        int token_id = token_d[row];
        
        embeddings_d[row][col] = wte_d[token_id][col] + wpe_d[row][col];
    }

}


//TODO - take those fixes and add them to the embedding benchmarks which I will do in the future.
extern "C" void embeddings_cuda(float (*wte_d)[d_model],
                                     float (*wpe_d)[d_model],
                                     int *token_d,
                                     float (*embeddings_d)[d_model],
                                     int token_length) {
    // 1. Define Block Size (Threads per block)
    // 16x16 = 256 threads per block. This is a standard, safe size.
    dim3 threadsPerBlock;
    threadsPerBlock.x = 16; 
    threadsPerBlock.y = 16;
    threadsPerBlock.z = 1;
    // 2. Define Grid Size (Number of blocks)
    // We divide total size by block size and round up using ceiling division: (N + block - 1) / block
    dim3 numBlocks;
    numBlocks.x = (d_model + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (token_length + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;


    printf("Launching Kernel with Grid(%d, %d) and Block(%d, %d)\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    // 3. Launch Kernel
    embedding_kernel<<<numBlocks, threadsPerBlock>>>(wte_d, wpe_d, token_d, embeddings_d, token_length);
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
}
