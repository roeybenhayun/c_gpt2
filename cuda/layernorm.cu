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
    // Enough for 1024 threads. "volatile" prevents the compiler from optimizing away memory reads during reduction
    __shared__ float s_data[THREADS_PER_BLOCK];

    // Calculate Identity
    int tid = threadIdx.x;      // My column (0..1023)
    int row_idx = blockIdx.x;   // My token (0..511)

    // Safety: If we have more blocks than rows, exit
    if (row_idx >= n_tokens) return;

    // --- PHASE 1: Calculate MEAN ---

    // 2. Load Data from Global to Register & Shared
    // Padding logic: If tid >= 768, we load 0.0 so we don't mess up the sum
    float val = 0.0f;
    if (tid < d_model) {
        val = input[row_idx][tid]; // Read Global
    }
    s_data[tid] = val; // Write Shared

    __syncthreads(); // Wait for everyone to fill shared mem

    // 3. Tree Reduction (Summation)
    // We fold the array in half: 1024 -> 512 -> 256 ... -> 1
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // 4. Thread 0 Calculates Mean
    // Result is now in s_data[0]
    if (tid == 0) {
        s_data[0] = s_data[0] / (float)d_model; 
    }
    __syncthreads(); // Wait for Thread 0

    // 5. Broadcast Mean to Private Register
    // CRITICAL: We save this in a register because s_data will be overwritten in Phase 2!
    float mean = s_data[0];


    // --- PHASE 2: Calculate VARIANCE ---

    // 6. Calculate Squared Difference
    // Use the 'val' we kept in our register from the start
    float diff_sq = 0.0f;
    if (tid < d_model) {
        float diff = val - mean;
        diff_sq = diff * diff;
    }
    
    // 7. Overwrite Shared Memory
    s_data[tid] = diff_sq; 
    
    __syncthreads();

    // 8. Tree Reduction (Summation of Diff Squared)
    // Same loop structure as before
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // 9. Thread 0 Calculates Variance (and std dev)
    // We put the inverse std dev back in s_data[0] for efficiency
    if (tid == 0) {
        float variance = s_data[0] / (float)d_model;
        // Pre-calculate the division factor: 1 / sqrt(var + eps)
        s_data[0] = rsqrtf(variance + eps); 
    }
    __syncthreads();

    // 10. Broadcast Inverse Std Dev
    float inv_std_dev = s_data[0];


    // --- PHASE 3: Final Normalize & Store ---
    
    // 11. Final Math
    // Only real columns write back (ignore the padding threads)
    if (tid < d_model) {
        // (x - mean) / std_dev
        float normalized = (val - mean) * inv_std_dev;

        // Apply Learnable Parameters (Gamma * norm + Beta)
        float result = normalized * gamma[tid] + beta[tid];

        // Write to Global Output
        output[row_idx][tid] = result;
    }
}


extern "C" void layernorm_cuda(float (*input)[d_model],int n_tokens,int d_model_size,float*gamma, float*beta,float (*output)[d_model], float eps ){
    dim3 threadsPerBlock;
    threadsPerBlock.x = d_model_size;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    
    // 2. Define Grid Size (Number of blocks)
    // We divide total size by block size and round up using ceiling division: (N + block - 1) / block
    dim3 numBlocks;
    numBlocks.x = n_tokens;
    numBlocks.y = 1;
    numBlocks.z = 1;

    printf("Launching Kernel with Grid(%d, %d) and Block(%d, %d)\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    layernorm_kernel<<<numBlocks, threadsPerBlock>>>(input,gamma,beta,output, n_tokens,eps);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

}