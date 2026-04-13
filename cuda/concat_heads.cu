#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/cuda_kernels.h"

__global__ void concat_heads_kernel(float *src, float *dest, int token_index, int _nof_heads,int _head_dim, int _ctx_len ) {

    // Map blockIdx.x to the head index
    int h = blockIdx.x;

    // MAp threadIdx.x to the dim index within the head
    int d = threadIdx.x;
    
    if (h < _nof_heads && d < _head_dim) {
            // 1. Calculate the flat index for the source: [h][token_idx][d]
            int src_idx = (h * _ctx_len * _head_dim) + (token_index * _head_dim) + d;
            
            // 2. Calculate the flat index for the destination: [token_idx][h * head_dim + d]
            int dest_idx = (token_index * (_nof_heads * _head_dim)) + (h * _head_dim) + d;
            
            // 3. Move the float!
            dest[dest_idx] = src[src_idx];
        }
}


extern "C" void concat_heads_cuda(float *src, float *dest, int token_index, int _nof_heads,int _head_dim, int _ctx_len ) {
    dim3 threadsPerBlock;
    threadsPerBlock.x = _head_dim; 
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    // 2. Define Grid Size (Number of blocks)
    // We divide total size by block size and round up using ceiling division: (N + block - 1) / block
    dim3 numBlocks;
    numBlocks.x = _nof_heads;
    numBlocks.y = 1;
    numBlocks.z = 1;


    //printf("Launching Kernel with Grid(%d, %d) and Block(%d, %d)\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    // 3. Launch Kernel
    concat_heads_kernel<<<numBlocks, threadsPerBlock>>>(src,dest,token_index,_nof_heads,_head_dim,_ctx_len);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "FATAL concat_heads_cuda launch: %s (token_index=%d nof_heads=%d head_dim=%d ctx_len=%d)\n",
                cudaGetErrorString(err), token_index, _nof_heads, _head_dim, _ctx_len);
        abort();
    }
}
