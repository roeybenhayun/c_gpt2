#include <stdio.h>
#include <stdlib.h>

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


    //printf("Launching Kernel with Grid(%d, %d) and Block(%d, %d)\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    // Pre-sync: surface any sticky error BEFORE blaming this kernel
    {
        cudaError_t pre = cudaDeviceSynchronize();
        if (pre != cudaSuccess) {
            fprintf(stderr, "FATAL embeddings_cuda PRE-sync (inherited error): %s (token_length=%d)\n",
                    cudaGetErrorString(pre), token_length);
            abort();
        }
    }

    // Verify all pointers are valid device pointers
    {
        cudaPointerAttributes attr;
        void *ptrs[4] = { (void*)wte_d, (void*)wpe_d, (void*)token_d, (void*)embeddings_d };
        const char *names[4] = { "wte_d", "wpe_d", "token_d", "embeddings_d" };
        for (int k = 0; k < 4; k++) {
            cudaError_t e = cudaPointerGetAttributes(&attr, ptrs[k]);
            if (e != cudaSuccess) {
                fprintf(stderr, "FATAL embeddings_cuda: cudaPointerGetAttributes(%s=%p) failed: %s\n",
                        names[k], ptrs[k], cudaGetErrorString(e));
                abort();
            }
            if (attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged) {
                fprintf(stderr, "FATAL embeddings_cuda: %s=%p is not device memory (type=%d)\n",
                        names[k], ptrs[k], (int)attr.type);
                abort();
            }
        }
    }

    // 3. Launch Kernel
    embedding_kernel<<<numBlocks, threadsPerBlock>>>(wte_d, wpe_d, token_d, embeddings_d, token_length);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "FATAL embeddings_cuda launch: %s (token_length=%d)\n",
                cudaGetErrorString(err), token_length);
        abort();
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "FATAL embeddings_cuda sync: %s (token_length=%d)\n",
                cudaGetErrorString(err), token_length);
        abort();
    }
}
