#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/cuda_kernels.h"

__global__ void softmax_kernel(float *in, float *out, int rows, int cols, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < rows) {
        float *row_ptr = in + i * stride;
        float *out_ptr = out + i * stride;

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

extern "C" void softmax_cuda(float *a, int a_r, int a_c, int stride, float *c_out) {
    // Use 1D block. 256 is a standard choice.
    dim3 threadsPerBlock(256, 1, 1);
    
    // Calculate blocks needed based on the actual number of rows (a_r)
    dim3 numBlocks((a_r + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

    // Launch Kernel
    softmax_kernel<<<numBlocks, threadsPerBlock>>>(a, c_out, a_r, a_c, stride);
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

    // Usually, we don't synchronize here to allow overlap with other GPU work.
    // But for a simple port, we can keep it if needed.
    cudaDeviceSynchronize();
}
