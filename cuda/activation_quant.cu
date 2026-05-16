/* Per-token dynamic INT8 activation quantization.
 *
 * One block per token. Threads cooperate to find amax over the row, then
 * write back INT8 values using scale = amax / 127. The output goes into
 * the cuBLAS INT8 GEMM as the activation (X) operand; scale_X is consumed
 * by the dequant+bias kernel that follows the GEMM.
 *
 * Convention matches the offline tool (tools/offline_quant/src/quantizer.py):
 *   scale = amax / 127  (symmetric; -128 unused to avoid abs-overflow)
 *   q     = clamp(round(x / scale), -127, 127)
 *
 * Compiled only under USE_INT8 — the host-side stub in gpt2.c is also
 * gated, so non-INT8 builds neither emit nor link this code.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/cuda_kernels.h"

#if defined(USE_INT8)

#define QUANT_BLOCK_SIZE   256          /* 8 warps; good occupancy for d up to ~8K */
#define QUANT_MAX_WARPS    (QUANT_BLOCK_SIZE / 32)
#define INT8_QUANT_MAX     127

__device__ __forceinline__ float warp_reduce_max(float v) {
    /* Intra-warp tree reduction via shuffle. After this returns, lane 0
     * of the warp holds the max over all 32 lanes; other lanes hold partial
     * results that the caller should discard. */
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}

__global__ void per_token_quant_kernel(
    const act_t *X,       /* [tokens, d] BF16 row-major */
    qweight_t   *X_q,     /* [tokens, d] INT8 row-major */
    qscale_t    *scale_X, /* [tokens]    FP32 one scale per row */
    int d)
{
    const int row = blockIdx.x;
    const act_t *x_row  = X   + (size_t)row * d;
    qweight_t   *xq_row = X_q + (size_t)row * d;

    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;
    /* Number of warps actually in use (covers non-multiple-of-32 blockDim, even
     * though our launch uses QUANT_BLOCK_SIZE which is a multiple of 32). */
    const int num_warps = (blockDim.x + 31) >> 5;

    /* ---- Phase 1: per-thread local amax over a stride-blockDim slice ---- */
    float local_max = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(to_float(x_row[i])));
    }

    /* ---- Phase 2: intra-warp reduction ---- */
    local_max = warp_reduce_max(local_max);

    /* ---- Phase 3: cross-warp combine via shared memory ---- */
    __shared__ float warp_maxes[QUANT_MAX_WARPS];
    if (lane == 0) warp_maxes[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < num_warps) ? warp_maxes[lane] : 0.0f;
        v = warp_reduce_max(v);
        if (lane == 0) warp_maxes[0] = v;
    }
    __syncthreads();

    const float amax = warp_maxes[0];

    /* ---- Phase 4: compute scale once, broadcast, write INT8 row ----
     * Zero-amax row → scale=1.0 produces all zeros, matching the offline
     * tool's zero-row handling. */
    const float scale     = (amax > 0.0f) ? (amax / (float)INT8_QUANT_MAX) : 1.0f;
    const float inv_scale = 1.0f / scale;
    if (tid == 0) scale_X[row] = scale;

    for (int i = tid; i < d; i += blockDim.x) {
        int q = __float2int_rn(to_float(x_row[i]) * inv_scale);
        /* Clamp belt-and-suspenders; with scale = amax/127 the value should
         * already fit, but rounding can produce ±128 at the boundary. */
        if (q >  INT8_QUANT_MAX) q =  INT8_QUANT_MAX;
        if (q < -INT8_QUANT_MAX) q = -INT8_QUANT_MAX;
        xq_row[i] = (qweight_t)q;
    }
}

extern "C" void per_token_quant_cuda(
    const act_t *X, qweight_t *X_q, qscale_t *scale_X,
    int tokens, int d)
{
    if (tokens <= 0) return;

    dim3 threads(QUANT_BLOCK_SIZE, 1, 1);
    dim3 blocks((unsigned)tokens, 1, 1);
    per_token_quant_kernel<<<blocks, threads>>>(X, X_q, scale_X, d);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "FATAL per_token_quant_cuda launch: %s (tokens=%d d=%d)\n",
                cudaGetErrorString(err), tokens, d);
        abort();
    }
}

#endif /* USE_INT8 */
