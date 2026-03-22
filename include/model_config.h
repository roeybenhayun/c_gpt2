#pragma once

#define vocab_size (50257)
#define ctx_len (1024)

#if !defined(GPT2_SMALL_MODEL) && !defined(GPT2_MEDIUM_MODEL) && !defined(GPT2_LARGE_MODEL)
    #define GPT2_MEDIUM_MODEL
#endif

#ifdef GPT2_SMALL_MODEL
    #define d_model (768)
    #define num_layers (12)
    #define nof_heads (12)
    #define MODEL_WEIGHTS_FILENAME "weights/gpt2_c_weights.bin"
    #define GPT2_PERFORMANCE_JSON_FILE_NAME "./logs/gpt2_small_performance.json"
    #define MODEL "gpt2_small"
#elif defined(GPT2_MEDIUM_MODEL)
    #define d_model (1024)
    #define num_layers (24)
    #define nof_heads (16)
    #define MODEL_WEIGHTS_FILENAME "weights/gpt2_medium_c_weights.bin"
    #define GPT2_PERFORMANCE_JSON_FILE_NAME "./logs/gpt2_medium_performance.json"
    #define MODEL "gpt2_medium"
#elif defined(GPT2_LARGE_MODEL)
    #define d_model (1280)
    #define num_layers (36)
    #define nof_heads (20)
    #define MODEL_WEIGHTS_FILENAME "weights/gpt2_large_c_weights.bin"
    #define GPT2_PERFORMANCE_JSON_FILE_NAME "./logs/gpt2_large_performance.json"
    #define MODEL "gpt2_large"
#else
    #error "No GPT-2 model size defined!"
#endif

#define head_dim (d_model / nof_heads)
#define d_ff (d_model * 4)
