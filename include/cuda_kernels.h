#pragma once

#include "model_config.h"

#ifdef __cplusplus
extern "C" {
#endif

void embeddings_cuda(float (*wte_d)[d_model],
                          float (*wpe_d)[d_model],
                          int *token_d,
                          float (*embeddings_d)[d_model],
                          int token_length);
                          

void layernorm_cuda(float (*input)[d_model],
                    int n_tokens,
                    int d_model_size,
                    float*gamma, 
                    float*beta,
                    float (*output)[d_model],
                    float eps
                );

#ifdef __cplusplus
}
#endif
