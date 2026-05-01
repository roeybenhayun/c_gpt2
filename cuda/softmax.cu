#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/cuda_kernels.h"

#define THREADS_PER_BLOCK 256

// One block per row — threads cooperate via shared memory reduction.
//
// Causal masking is folded into this kernel via the `causal_mask` flag.
// Previously the prefill path did:
//     casual_masking_cuda(...)   // wrote -INFINITY into the upper triangle
//     softmax_cuda(...)
// That separate kernel doubled in cost under BF16 (the divergent per-row
// store loop didn't translate well to half-precision stores) and was the
// dominant remaining BF16 prefill penalty. Folding it here means:
//   * no extra global-memory pass to write the mask
//   * no extra kernel launch per (layer × head)
//   * masking is FP32-arithmetic, so no dtype-specific behavior
// When `causal_mask` is true, any column index j > row_idx is treated as
// -INFINITY in phase 1 (max) and 0 in phase 2 (exp+sum), and the output at
// those positions is written as 0. Otherwise the kernel behaves exactly
// as before.
__global__ void softmax_kernel(act_t *in, act_t *out, int rows, int cols, int stride, float temperature, int causal_mask) {
    __shared__ float s_data[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int row_idx = blockIdx.x;

    if (row_idx >= rows) return;

    float inv_temp = 1.0f / temperature;
    act_t *row_ptr = in + row_idx * stride;
    act_t *out_ptr = out + row_idx * stride;

    // Effective last valid column for this row. Without masking, the row
    // can attend to all cols; with causal masking, only j <= row_idx.
    int last_col = causal_mask ? row_idx : (cols - 1);

    // --- PHASE 1: Find max (parallel reduction) ---
    float local_max = -INFINITY;
    for (int j = tid; j < cols; j += blockDim.x) {
        if (j > last_col) continue;  // masked out — skip
        float val = to_float(row_ptr[j]) * inv_temp;
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
        if (j > last_col) {
            // Masked position: write 0 directly, contribute nothing to the sum.
            out_ptr[j] = to_act(0.0f);
            continue;
        }
        float e = expf(to_float(row_ptr[j]) * inv_temp - row_max);
        out_ptr[j] = to_act(e);
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
        if (j > last_col) continue;  // masked positions already written as 0
        out_ptr[j] = to_act(to_float(out_ptr[j]) * inv_sum);
    }
}

extern "C" void softmax_cuda(act_t *a, int a_r, int a_c, int stride, act_t *c_out, float temperature, int causal_mask) {
    dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);
    dim3 numBlocks(a_r, 1, 1);

    softmax_kernel<<<numBlocks, threadsPerBlock>>>(a, c_out, a_r, a_c, stride, temperature, causal_mask);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "FATAL softmax_cuda launch: %s (a_r=%d a_c=%d stride=%d causal_mask=%d)\n",
                cudaGetErrorString(err), a_r, a_c, stride, causal_mask);
        abort();
    }
}
