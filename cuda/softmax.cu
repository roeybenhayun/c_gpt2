#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/cuda_kernels.h"

#define THREADS_PER_BLOCK 256

// One block per row — threads cooperate via shared memory reduction
__global__ void softmax_kernel(float *in, float *out, int rows, int cols, int stride, float temperature) {
    __shared__ float s_data[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int row_idx = blockIdx.x;

    if (row_idx >= rows) return;

    float inv_temp = 1.0f / temperature;
    float *row_ptr = in + row_idx * stride;
    float *out_ptr = out + row_idx * stride;

    // --- PHASE 1: Find max (parallel reduction) ---
    float local_max = -INFINITY;
    for (int j = tid; j < cols; j += blockDim.x) {
        float val = row_ptr[j] * inv_temp;
        if (val > local_max) local_max = val;
    }
    s_data[tid] = local_max;
    __syncthreads();

    // Tree reduction for max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_data[tid + s] > s_data[tid])
                s_data[tid] = s_data[tid + s];
        }
        __syncthreads();
    }
    float row_max = s_data[0];
    __syncthreads();

    // --- PHASE 2: Sum exponentials (parallel reduction) ---
    float local_sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float e = expf(row_ptr[j] * inv_temp - row_max);
        out_ptr[j] = e;
        local_sum += e;
    }
    s_data[tid] = local_sum;
    __syncthreads();

    // Tree reduction for sum
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    float sum_exp = s_data[0];
    __syncthreads();

    // --- PHASE 3: Normalize ---
    float inv_sum = 1.0f / sum_exp;
    for (int j = tid; j < cols; j += blockDim.x) {
        out_ptr[j] *= inv_sum;
    }
}

extern "C" void softmax_cuda(float *a, int a_r, int a_c, int stride, float *c_out, float temperature) {
    dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);
    dim3 numBlocks(a_r, 1, 1);

    softmax_kernel<<<numBlocks, threadsPerBlock>>>(a, c_out, a_r, a_c, stride, temperature);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "FATAL softmax_cuda launch: %s (a_r=%d a_c=%d stride=%d)\n",
                cudaGetErrorString(err), a_r, a_c, stride);
        abort();
    }
}
