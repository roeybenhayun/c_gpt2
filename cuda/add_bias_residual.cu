#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/cuda_kernels.h"

#define THREADS_PER_BLOCK 256

/* out = residual + (x + bias), fusing the add_bias_cuda + add_2d_cuda pair.
 *
 * Both transformer residual joins ran as two kernels: add_bias wrote the biased
 * projection back to global memory, then add_2d read it straight back in to add
 * the residual. Two launches and one redundant round trip through HBM per join,
 * twice per layer -- 144 launches per token on Large for what is a single pass.
 */
__global__ void add_bias_residual_kernel(const act_t *x, const weight_t *bias,
                                         const act_t *residual, act_t *out,
                                         int n_rows, int n_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows * n_cols) return;

    int col = idx % n_cols;
    float val = to_float(x[idx]) + to_float(bias[col]) + to_float(residual[idx]);
    out[idx] = to_act(val);
}

void add_bias_residual_cuda(const act_t *x, const weight_t *bias,
                            const act_t *residual, act_t *out,
                            int n_rows, int n_cols) {
    int total = n_rows * n_cols;
    int threadsPerBlock = THREADS_PER_BLOCK;
    int numBlocks = (total + threadsPerBlock - 1) / threadsPerBlock;
    add_bias_residual_kernel<<<numBlocks, threadsPerBlock>>>(x, bias, residual, out,
                                                             n_rows, n_cols);
}
