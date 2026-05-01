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

#if defined(USE_BF16)
    #ifdef __CUDACC__
        #include <cuda_bf16.h>
        typedef __nv_bfloat16 act_t;
        typedef __nv_bfloat16 weight_t;
    #else
        typedef __bf16 act_t;
        typedef __bf16 weight_t;
    #endif
    #define DTYPE_NAME "bf16"
#elif defined(USE_FP16)
    #ifdef __CUDACC__
        #include <cuda_fp16.h>
        typedef __half act_t;
        typedef __half weight_t;
    #else
        typedef _Float16 act_t;
        typedef _Float16 weight_t;
    #endif
    #define DTYPE_NAME "fp16"
#else
    typedef float act_t;
    typedef float weight_t;
    #define DTYPE_NAME "fp32"
#endif

// Device-side narrow<->float helpers. In FP32 builds these are identity
// and inline to nothing; in BF16/FP16 builds they emit a single conversion
// instruction. Use them on every global-memory read/write inside kernels.
#ifdef __CUDACC__
__device__ __forceinline__ float to_float(act_t x) {
#if defined(USE_BF16)
    return __bfloat162float(x);
#elif defined(USE_FP16)
    return __half2float(x);
#else
    return x;
#endif
}
__device__ __forceinline__ act_t to_act(float x) {
#if defined(USE_BF16)
    return __float2bfloat16(x);
#elif defined(USE_FP16)
    return __float2half(x);
#else
    return x;
#endif
}
#endif
