#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#include "../include/cuda_kernels.h"

__global__ void embedding_kernel(float (*wte_d)[d_model],
                                 float (*wpe_d)[d_model],
                                 int *token_d,
                                 float (*embeddings_d)[d_model],
                                 int start_row,
                                 int n_rows) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int local_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (local_row < n_rows && col < d_model) {
        int row = start_row + local_row;
        int token_id = token_d[row];
        embeddings_d[row][col] = wte_d[token_id][col] + wpe_d[row][col];
    }

}


//TODO - take those fixes and add them to the embedding benchmarks which I will do in the future.
extern "C" void embeddings_cuda(float (*wte_d)[d_model],
                                     float (*wpe_d)[d_model],
                                     int *token_d,
                                     float (*embeddings_d)[d_model],
                                     int start_row,
                                     int n_rows) {
    dim3 threadsPerBlock;
    threadsPerBlock.x = 16;
    threadsPerBlock.y = 16;
    threadsPerBlock.z = 1;

    dim3 numBlocks;
    numBlocks.x = (d_model + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (n_rows + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;

    embedding_kernel<<<numBlocks, threadsPerBlock>>>(wte_d, wpe_d, token_d, embeddings_d, start_row, n_rows);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "FATAL embeddings_cuda launch: %s (start_row=%d n_rows=%d)\n",
                cudaGetErrorString(err), start_row, n_rows);
        abort();
    }
}
