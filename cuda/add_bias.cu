
#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/cuda_kernels.h"



__global__ void add_bias_kernel(float *a, float *b, int a_r, int a_c) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;
        if (row < a_r && col < a_c) {
            a[row * a_c + col] += b[col];
        }
}


//float *a, int a_r, int a_c, float *b, float *out)

extern "C" void add_bias_cuda(float *a, int a_r, int a_c, float *b)
{
                                     
    // 1. Define Block Size (Threads per block)
    // 16x16 = 256 threads per block. This is a standard, safe size.
    dim3 threadsPerBlock;
    threadsPerBlock.x = 16; 
    threadsPerBlock.y = 16;
    threadsPerBlock.z = 1;
    // 2. Define Grid Size (Number of blocks)
    // We divide total size by block size and round up using ceiling division: (N + block - 1) / block
    dim3 numBlocks;
    numBlocks.x = (a_c + 15) / 16;
    numBlocks.y = (a_r + 15) / 16;
    numBlocks.z = 1;


    printf("Launching Kernel with Grid(%d, %d) and Block(%d, %d)\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    // 3. Launch Kernel
    add_bias_kernel<<<numBlocks, threadsPerBlock>>>(a, b, a_r, a_c);
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
}
