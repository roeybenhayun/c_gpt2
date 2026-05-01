#pragma once

#include "model_config.h"

#ifdef __cplusplus
extern "C" {
#endif

void embeddings_cuda(weight_t (*wte_d)[d_model],
                          weight_t (*wpe_d)[d_model],
                          int *token_d,
                          act_t (*embeddings_d)[d_model],
                          int start_row,
                          int n_rows);


void layernorm_cuda(act_t (*input)[d_model],
                    int n_tokens,
                    int d_model_size,
                    weight_t *gamma,
                    weight_t *beta,
                    act_t (*output)[d_model],
                    float eps
                );


void add_bias_cuda
                (act_t *a,
                    int a_r,
                    int a_c,
                    weight_t *b,
                    act_t *out);

// `causal_mask` (0/1): when set, treat j > row_idx as -INFINITY internally,
// replacing the separate casual_masking_cuda kernel that used to run before
// softmax in the prefill path.
void softmax_cuda
                (act_t *a,
                    int a_r,
                    int a_c,
                    int stride,
                    act_t *c_out,
                    float temperature,
                    int causal_mask
                );


void casual_masking_cuda(act_t *in,
                                     int stride,
                                     int tokens);

void gelu_cuda(act_t *in, int cols, int rows, act_t *out);


void concat_heads_cuda(act_t *src, act_t *dest, int token_index, int _nof_heads,int _head_dim, int _ctx_len );

void add_2d_cuda(act_t *a, int a_r, int a_c, act_t *b, act_t * out);


#ifdef __cplusplus
}
#endif
