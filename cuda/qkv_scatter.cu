#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/cuda_kernels.h"

#define THREADS_PER_BLOCK 256

/* Adds the fused QKV bias and scatters the fused projection into its three
 * destinations.
 *
 * The Q/K/V projections used to be three separate GEMMs followed by three
 * add_bias launches — 6 launches per layer, 216 per token on Large, with the
 * three [d_model x d_model] matrices running at only ~50% of peak bandwidth
 * because they are individually too small. They are now one
 * [3*d_model x d_model] GEMM writing a packed [rows x 3*d_model] scratch,
 * followed by this single kernel: 2 launches per layer instead of 6.
 *
 * The scatter is needed because the three results do not live together: Q goes
 * to the Q buffer while K and V are appended to their respective KV caches, and
 * each destination has its own row stride.
 */
__global__ void qkv_bias_scatter_kernel(const act_t *qkv, const weight_t *bias,
                                        act_t *q_dst, act_t *k_dst, act_t *v_dst,
                                        int n_rows, int d, int dst_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_rows * 3 * d;
    if (idx >= total) return;

    int row = idx / (3 * d);
    int col = idx - row * (3 * d);

    float val = to_float(qkv[(size_t)row * (3 * d) + col]) + to_float(bias[col]);

    /* col < d -> Q, col < 2d -> K, else V; each destination is d wide. */
    act_t *dst;
    int out_col;
    if (col < d) {
        dst = q_dst; out_col = col;
    } else if (col < 2 * d) {
        dst = k_dst; out_col = col - d;
    } else {
        dst = v_dst; out_col = col - 2 * d;
    }

    dst[(size_t)row * dst_stride + out_col] = to_act(val);
}

void qkv_bias_scatter_cuda(const act_t *qkv, const weight_t *bias,
                           act_t *q_dst, act_t *k_dst, act_t *v_dst,
                           int n_rows, int d, int dst_stride) {
    int total = n_rows * 3 * d;
    int threadsPerBlock = THREADS_PER_BLOCK;
    int numBlocks = (total + threadsPerBlock - 1) / threadsPerBlock;
    qkv_bias_scatter_kernel<<<numBlocks, threadsPerBlock>>>(qkv, bias, q_dst, k_dst, v_dst,
                                                            n_rows, d, dst_stride);
}
