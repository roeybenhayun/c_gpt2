
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/cuda_kernels.h"



__global__ void add_2d_kernel(float *a, float *b, float *out, int a_r, int a_c) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < a_r && col < a_c) {
        int idx = row * a_c + col;
        out[idx] = a[idx] + b[idx];
    }
}

extern "C" void add_2d_cuda(float *a, int a_r, int a_c, float *b, float *out)
{
    if (out == NULL) {
        out = a;
    }
                                     
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


    //printf("Launching Kernel with Grid(%d, %d) and Block(%d, %d)\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    // 3. Launch Kernel
    add_2d_kernel<<<numBlocks, threadsPerBlock>>>(a, b, out,a_r, a_c);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "FATAL add_2d_cuda launch: %s (a_r=%d a_c=%d)\n",
                cudaGetErrorString(err), a_r, a_c);
        abort();
    }
}
