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


void add_bias_cuda
                (float *a, 
                    int a_r, 
                    int a_c, 
                    float *b,
                    float *out);
            
void softmax_cuda
                (float *a, 
                    int a_r, 
                    int a_c, 
                    int stride, 
                    float *c_out,
                    float temperature
                );


void casual_masking_cuda(float *in,
                                     int stride,
                                     int tokens);

void gelu_cuda(float *in, int cols, int rows, float *out);


void concat_heads_cuda(float *src, float *dest, int token_index, int _nof_heads,int _head_dim, int _ctx_len );

void add_2d_cuda(float *a, int a_r, int a_c, float *b, float * out);


#ifdef __cplusplus
}
#endif
