#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/cuda_kernels.h"

__global__ void softmax_kernel(float *in, float *out, int rows, int cols, int stride,float temperature) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if (i < rows) {
        float inv_temp = 1.0f / temperature;
        float *row_ptr = in + i * stride;
        float *out_ptr = out + i * stride;

        // 1. Find Max
        float row_max = -INFINITY;
        for (int j = 0; j < cols; j++) {
            float val = row_ptr[j] * inv_temp;
            if (val > row_max) row_max = val;
        }

        // 2. Sum Exponentials
        float sum_exp = 0.0f;
        for (int j = 0; j < cols; j++) {
            float val = (row_ptr[j] * inv_temp) - row_max;
            float e = expf(val);
            out_ptr[j] = e;
            sum_exp += e;
        }

        // 3. Normalize
        for (int j = 0; j < cols; j++) {
            out_ptr[j] /= sum_exp;
        }
    }
}

extern "C" void softmax_cuda(float *a, int a_r, int a_c, int stride, float *c_out,float temperature) {
    // Use 1D block. 256 is a standard choice.
    dim3 threadsPerBlock(256, 1, 1);
    
    // Calculate blocks needed based on the actual number of rows (a_r)
    dim3 numBlocks((a_r + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

    // Launch Kernel
    softmax_kernel<<<numBlocks, threadsPerBlock>>>(a, c_out, a_r, a_c, stride,temperature);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "FATAL softmax_cuda launch: %s (a_r=%d a_c=%d stride=%d)\n",
                cudaGetErrorString(err), a_r, a_c, stride);
        abort();
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "FATAL softmax_cuda sync: %s (a_r=%d a_c=%d stride=%d)\n",
                cudaGetErrorString(err), a_r, a_c, stride);
        abort();
    }
}
