#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/cuda_kernels.h"


__global__ void gelu_kernel(float *in, int cols, int rows) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float term = 0.79788456f; // this is fixed value. move it to defines 
     
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        float val = in[idx];        
        in[idx] =  0.5f * val * (1.0f + tanhf (term * (val + 0.044715f*val*val*val)));
    }
}

extern "C" void gelu_cuda(float *in, int cols, int rows, float *out){
    
    dim3 threadsPerBlock;
    threadsPerBlock.x = 32; 
    threadsPerBlock.y = 32;
    threadsPerBlock.z = 1;
    
    dim3 numBlocks;
    numBlocks.x = (cols + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (rows + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;

    // printf("Launching Kernel with Grid(%d, %d) and Block(%d, %d)\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    // 3. Launch Kernel
    gelu_kernel<<<numBlocks, threadsPerBlock>>>(in, cols, rows);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "FATAL gelu_cuda launch: %s (rows=%d cols=%d)\n",
                cudaGetErrorString(err), rows, cols);
        abort();
    }
}