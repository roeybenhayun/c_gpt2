#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/cuda_kernels.h"

__global__ void casual_masking_kernel(float *in, int stride, int tokens) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < tokens) {
        float *row_ptr = in + i * stride;
        for (int j = i+1; j < tokens ; j++) {
            row_ptr[j] = -INFINITY;
        }
    }
}


extern "C" void casual_masking_cuda(float *in,
                                     int stride,
                                     int tokens) {
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


    //printf("Launching Kernel with Grid(%d, %d) and Block(%d, %d)\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    // 3. Launch Kernel
    casual_masking_kernel<<<numBlocks, threadsPerBlock>>>(in,stride,tokens);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "FATAL casual_masking_cuda launch: %s (stride=%d tokens=%d)\n",
                cudaGetErrorString(err), stride, tokens);
        abort();
    }
}
