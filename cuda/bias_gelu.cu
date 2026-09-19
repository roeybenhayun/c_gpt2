#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/cuda_kernels.h"

#define THREADS_PER_BLOCK 256

/* x = gelu(x + bias), fusing the add_bias_cuda + gelu_cuda pair on the MLP's
 * first projection.
 *
 * add_bias wrote the biased [rows x d_ff] activation back to global memory and
 * gelu read it straight back in. d_ff is 4*d_model -- the widest activation in
 * the layer -- so that was the largest redundant round trip left, plus an extra
 * launch per layer (36 per token on Large).
 *
 * The tanh approximation matches gelu_kernel exactly so results are unchanged.
 */
__global__ void bias_gelu_kernel(act_t *x, const weight_t *bias,
                                 int n_rows, int n_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows * n_cols) return;

    const float term = 0.79788456f;
    int col = idx % n_cols;

    float val = to_float(x[idx]) + to_float(bias[col]);
    float result = 0.5f * val * (1.0f + tanhf(term * (val + 0.044715f*val*val*val)));
    x[idx] = to_act(result);
}

void bias_gelu_cuda(act_t *x, const weight_t *bias, int n_rows, int n_cols) {
    int total = n_rows * n_cols;
    int threadsPerBlock = THREADS_PER_BLOCK;
    int numBlocks = (total + threadsPerBlock - 1) / threadsPerBlock;
    bias_gelu_kernel<<<numBlocks, threadsPerBlock>>>(x, bias, n_rows, n_cols);
}
