#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/cuda_kernels.h"

#define THREADS_PER_BLOCK 1024 

__global__ void layernorm_kernel(float (*input)[d_model], 
                                 float *gamma, 
                                 float *beta, 
                                 float (*output)[d_model], 
                                 int n_tokens,
                                float eps) {
    
    // 1. Setup Shared Memory
    __shared__ float s_data[THREADS_PER_BLOCK];

    int tid = threadIdx.x;      // My thread ID (0..1023)
    int row_idx = blockIdx.x;   // My token row

    // Safety: If we have more blocks than rows, exit
    if (row_idx >= n_tokens) return;

    // --- PHASE 1: Calculate MEAN ---
    // Use a stride loop to sum all elements in the row, even if row > 1024
    float sum = 0.0f;
    for (int i = tid; i < d_model; i += blockDim.x) {
        sum += input[row_idx][i];
    }
    s_data[tid] = sum;
    __syncthreads();

    // Tree Reduction (Summation)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 holds the total sum; broadcast the mean to all threads via a register
    float mean = s_data[0] / (float)d_model;
    __syncthreads(); 


    // --- PHASE 2: Calculate VARIANCE ---
    float diff_sq_sum = 0.0f;
    for (int i = tid; i < d_model; i += blockDim.x) {
        float diff = input[row_idx][i] - mean;
        diff_sq_sum += diff * diff;
    }
    s_data[tid] = diff_sq_sum; 
    __syncthreads();

    // Tree Reduction (Summation of Diff Squared)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 calculates Inverse Standard Deviation
    if (tid == 0) {
        float variance = s_data[0] / (float)d_model;
        s_data[0] = rsqrtf(variance + eps); 
    }
    __syncthreads();

    float inv_std_dev = s_data[0];


    // --- PHASE 3: Final Normalize & Store ---
    for (int i = tid; i < d_model; i += blockDim.x) {
        float val = input[row_idx][i];
        float normalized = (val - mean) * inv_std_dev;
        output[row_idx][i] = normalized * gamma[i] + beta[i];
    }
}


extern "C" void layernorm_cuda(float (*input)[d_model],int n_tokens,int d_model_size,float*gamma, float*beta,float (*output)[d_model], float eps ){
    dim3 threadsPerBlock;
    threadsPerBlock.x = THREADS_PER_BLOCK;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    
    dim3 numBlocks;
    numBlocks.x = n_tokens;
    numBlocks.y = 1;
    numBlocks.z = 1;

    layernorm_kernel<<<numBlocks, threadsPerBlock>>>(input,gamma,beta,output, n_tokens,eps);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel Launch Error (layernorm): %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
}
