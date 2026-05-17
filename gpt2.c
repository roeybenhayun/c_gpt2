/// TODO
// * Change to loop to output token by token (instead of all at once)
// * Move the token selection logic into a separate function
// * CLI arg for temperature
// * CLI arg for top_k
// * CLI atg for top_p
// * Function to cleanup data structures
// * Update layer names consistently
// * Save ~147MB by using remove wte_T and use in dot_2d with transposed flag 

#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <jansson.h>

#ifdef USE_GPU
        #define USE_CUDA
#endif

#define SERVER_PORT 65432
#define SERVER_IP "127.0.0.1"

#include "include/model_config.h"



#ifdef USE_CUDA
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
    #include "include/cuda_kernels.h"
#else 
    #if defined  (USE_ACCELERATE)
        #include <Accelerate/Accelerate.h>
        #define CBLAS_ROW_MAJOR CblasRowMajor
        #define CBLAS_NO_TRANS CblasNoTrans
    #elif defined (USE_ACCELERATE_X86)
        #include <cblas.h>
    #endif
#endif


#ifdef USE_CUDA
#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t err__ = (call);                                            \
    if (err__ != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error at %s:%d in %s: %s\n",                 \
                __FILE__, __LINE__, #call, cudaGetErrorString(err__));     \
        exit(1);                                                           \
    }                                                                      \
} while (0)
#endif

//#ifdef USE_ACCELERATE
//#include <Accelerate/Accelerate.h>
//#define CBLAS_ROW_MAJOR CblasRowMajor
//#define CBLAS_NO_TRANS CblasNoTrans
//#endif

#define APPLY_ATTENTION_SCALING (1)
#define PI (3.14159265358979323846)

static void print_2d_tensor(char* name, act_t *a, int a_r_full_dim, int a_c_full_dim, int r_idx_to_print, int c_idx_to_print, int enable);
static float mean_(act_t *x, int len);
static float variance_(act_t *x,int len, float mean);
static void softmax_2d(act_t *a, int a_r, int a_c, int stride, act_t * c_out);
static void dot_2d(act_t *a, // Matrix A (input)
            int a_r, // a rows
            int a_c, // a column
            int lda, // leading dim a
            act_t *b, // Matrix B (input)
            int b_r, // b rows
            int b_c, // b columns
            int ldb, // leading dim b
            act_t * c_out, // Matrix C (output)
            int c_r, // c rows
            int c_c, // c columns
            int ldc, // stride c
            int transpose_b, //
            int apply_attention_scaling
        );
static void transpose_2d(act_t *a, int a_r, int a_c, act_t *b);
static void layernorm_2d(act_t *a, int a_r, int a_c, weight_t * ln_gamma, weight_t * ln_beta, act_t * out, float epsilon);
static float gelu(float x);
static void gelu_2d(act_t *a,int a_c, int a_r, act_t *out);
static void apply_casual_masking(act_t * a, int a_c,int size);


typedef struct {
    float prob;
    int index;
} TokenProb;

// Comparison function for qsort
// It sorts TokenProb structs in DESCENDING order of their 'prob' field.
int compareTokenProbs(const void *a, const void *b) {
    const TokenProb *tokenA = (const TokenProb *)a;
    const TokenProb *tokenB = (const TokenProb *)b;

    if (tokenA->prob < tokenB->prob) {
        return 1;  // Return positive if A's prob is smaller (A comes after B)
    }
    if (tokenA->prob > tokenB->prob) {
        return -1; // Return negative if A's prob is larger (A comes before B)
    }
    return 0;      // Probabilities are equal
}

/**
 * @brief Performs Top-K sampling from a given probability distribution.
 *
 * This function identifies the K most probable tokens, renormalizes their
 * probabilities, and provides them for subsequent sampling.
 *
 * @param probs               Pointer to the full probability distribution (from softmax).
 * @param vocab_size          The total size of the vocabulary.
 * @param k                   The number of top tokens to consider. If k <= 0 or k > vocab_size,
 * it defaults to vocab_size (equivalent to multinomial sampling).
 * @param top_k_indices_out   Output array to store the indices of the top K tokens.
 * Must be pre-allocated by the caller to a size of at least 'k'.
 * @param top_k_probs_out     Output array to store the renormalized probabilities of the top K tokens.
 * Must be pre-allocated by the caller to a size of at least 'k'.
 */
void top_k_sample(act_t *probs, int v_size, int k,
                  int *top_k_indices_out, act_t *top_k_probs_out) {

    // Ensure k is within a valid range. If k is too large, it defaults to multinomial sampling.
    if (k <= 0 || k > v_size) {
        k = v_size;
    }

    // Declare a local (stack-allocated) array of TokenProb structs.
    // This is valid in C99 as a Variable Length Array (VLA) if vocab_size
    // is not a compile-time constant, but for large vocab_size (like 50257),
    // this can cause stack overflow. For robustness, global/static or dynamic
    // allocation (malloc) is usually preferred for very large arrays.
    // However, adhering to the "no dynamic allocation" constraint:
    TokenProb token_probs_list[v_size];

    // 1. Populate the list of (probability, original_index) pairs
    for (int i = 0; i < v_size; i++) {
        token_probs_list[i].prob = probs[i];
        token_probs_list[i].index = i;
    }

    // 2. Sort the list in descending order of probabilities
    // qsort modifies the array in-place.
    qsort(token_probs_list, v_size, sizeof(TokenProb), compareTokenProbs);

    // 3. Extract the top K tokens and their probabilities
    float sum_top_k_probs = 0.0f;
    for (int i = 0; i < k; i++) {
        top_k_indices_out[i] = token_probs_list[i].index;
        top_k_probs_out[i] = token_probs_list[i].prob;
        sum_top_k_probs += token_probs_list[i].prob;
    }

    // 4. Renormalize the probabilities of the top K tokens
    // This makes sure they sum to 1.0 again for accurate sampling.
    if (sum_top_k_probs > 0.0f) { // Avoid division by zero if all top K probabilities were 0
        for (int i = 0; i < k; i++) {
            top_k_probs_out[i] /= sum_top_k_probs;
        }
    } else {
        // Fallback: If sum_top_k_probs is 0 (e.g., all top K tokens had 0 probability),
        // distribute probability uniformly among them to allow sampling.
        // This scenario is rare with a proper softmax output.
        if (k > 0) {
            for (int i = 0; i < k; i++) {
                top_k_probs_out[i] = 1.0f / k;
            }
        }
    }
}

#if 0
static void add_tensor_to_layer(json_t *layer_obj, const char *tensor_name, act_t *a, int a_r, int a_c, int r_idx, int c_idx) {
    json_t *tensor_array = json_array();
    for (int i = 0; i < r_idx; i++) {
        json_t *row = json_array();
        for (int j = 0; j < c_idx; j++) {
            float val = *(a + i * a_c + j);
            json_array_append_new(row, json_real(val));
        }
        json_array_append_new(tensor_array, row);
    }
    json_object_set_new(layer_obj, tensor_name, tensor_array);
}
#endif


static void send_json_to_tokenizer(const char *json_str, char *response_buf, size_t buf_size) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("Socket creation failed");
        exit(1);
    }

    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(SERVER_PORT);
    inet_pton(AF_INET, SERVER_IP, &server_addr.sin_addr);

    if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("Connection to tokenizer server failed");
        close(sock);
        exit(1);
    }

    send(sock, json_str, strlen(json_str), 0);

    // Read in a loop until the peer closes the connection or the buffer fills.
    // Single-recv was capped at 4 KB, which truncated long token lists (~960
    // tokens easily exceeds 4 KB) and produced unparseable responses.
    size_t total = 0;
    while (total + 1 < buf_size) {
        int len = recv(sock, response_buf + total, buf_size - 1 - total, 0);
        if (len < 0) {
            perror("Receive failed");
            close(sock);
            exit(1);
        }
        if (len == 0) break;  // peer closed
        total += len;
    }
    response_buf[total] = '\0';

    close(sock);
}


static void softmax_2d(act_t *a, int a_r, int a_c,int stride, act_t * c_out){
    for (int i=0; i<a_r; i++){
        // 1. Find the maximum value in the current row
        float row_max = -INFINITY;
        for (int j=0; j<a_c; j++){
            if (*(a + i*stride + j) > row_max) {
                row_max = *(a + i*stride + j);
            }
        }
        // 2. Calculate the exponentials with the maximum subtracted and the sum
        float sum_exp = 0.0;
        for (int j = 0; j < a_c; j++) {
            float shifted = *(a + i * stride + j) - row_max;
            float exp_val = expf(shifted);
            *(c_out + i * stride + j) = exp_val;
            sum_exp += exp_val;
        }
        if (sum_exp == 0.0f && i == 1) {
            printf("WARNING: softmax sum_exp == 0 at row i=%d — input may be all -inf\n", i);
        }
        // 3. Normalize
        for (int j = 0; j < a_c; j++) {
            *(c_out + i * stride + j) /= sum_exp;
        }                 
    }
}

#ifdef USE_CUDA
// Helper function to initialize and get a static cuBLAS handle.
static cublasHandle_t get_cublas_handle() {
    static bool handle_initialized = false;
    static cublasHandle_t handle;
    if (!handle_initialized) {
        cublasStatus_t stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "FATAL: cuBLAS handle initialization failed\n");
            exit(1);
        }
        handle_initialized = true;
    }
    return handle;
}


// Input/output dtype for the GEMM. Compute type is always FP32
// (FP32 accumulator) regardless of build, matching the standard
// inference recipe.
#if defined(USE_BF16)
    #define GEMM_DATA_TYPE CUDA_R_16BF
#elif defined(USE_FP16)
    #define GEMM_DATA_TYPE CUDA_R_16F
#else
    #define GEMM_DATA_TYPE CUDA_R_32F
#endif

static void dot_2d_gpu(act_t *a,int a_r, int a_c, int lda, act_t *b,int b_r,int b_c,int ldb, act_t * c_out,int c_r, int c_c, int ldc, int transpose_b,int apply_attention_scaling ){
    // NOTE: Assumes 'a', 'b', and 'c_out' are pointers to GPU memory.
    cublasHandle_t handle = get_cublas_handle();

    float alpha = 1.0f;
    if (apply_attention_scaling) {
        alpha = 1.0f / sqrtf((float)a_c);
    }
    const float beta = 0.0f;

    const int M = a_r;
    const int K = a_c;
    const int N = transpose_b ? b_r : b_c;

    const cublasOperation_t opB = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t opA = CUBLAS_OP_N;

    cublasStatus_t stat = cublasGemmEx(
        handle, opB, opA, N, M, K,
        &alpha,
        b, GEMM_DATA_TYPE, ldb,
        a, GEMM_DATA_TYPE, lda,
        &beta,
        c_out, GEMM_DATA_TYPE, ldc,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr,
            "FATAL: cublasGemmEx failed with status %d\n"
            "  dims: a[%d x %d] lda=%d  b[%d x %d] ldb=%d  c[%d x %d] ldc=%d  trans_b=%d\n"
            "  cublas: M=%d N=%d K=%d\n",
            stat, a_r, a_c, lda, b_r, b_c, ldb, c_r, c_c, ldc, transpose_b, M, N, K);
        abort();
    }
}

#if defined(USE_INT8)
/* Forward declarations of the per-GEMM scratch buffers — defined in the
 * global device-state block below. dot_2d_gpu_int8 references them by
 * name, and that function appears earlier in the file (next to dot_2d_gpu)
 * for locality with the GEMM dispatch code. */
extern qweight_t *X_int8_scratch_d;
extern qscale_t  *scale_X_scratch_d;
extern int32_t   *Y_int32_scratch_d;

// INT8 weight-only GEMM dispatch (W8A8 via cuBLAS).
//
// Pipeline per call:
//   1. Quantize X (BF16 [M,K]) per-token → X_int8 [M,K] + scale_X [M]
//   2. cublasGemmEx in INT8: X_int8 @ W_int8^T → Y_int32 [M,N]
//      (A=W_int8 transposed, B=X_int8, compute=INT32, algo=TENSOR_OP)
//   3. Dequantize Y_int32 → Y_bf16 via scale_W ⊗ scale_X
//
// Bias is NOT applied here — caller still invokes add_bias_cuda. Phase 5
// will fuse dequant + bias and remove that trailing call.
//
// Shape contract matches dot_2d_gpu: X is [x_r=M, x_c=K], W is stored
// [w_r=N, w_c=K] (transposed at GEMM time), Y is [y_r=M, y_c=N].
static void dot_2d_gpu_int8(
    act_t *x_bf16,        int x_r, int x_c, int ldx,
    qweight_t *w_int8,    int w_r, int w_c, int ldw, qscale_t *scale_w,
    act_t *y_bf16,        int y_r, int y_c, int ldy)
{
    cublasHandle_t handle = get_cublas_handle();

    const int M = x_r;
    const int K = x_c;
    const int N = w_r;          /* W stored [out=w_r, in=w_c], so output dim N = w_r */

    /* 1. Per-token quantize X. */
    per_token_quant_cuda(x_bf16, X_int8_scratch_d, scale_X_scratch_d, M, K);

    /* 2. cuBLAS INT8 GEMM. alpha/beta are int32 for CUBLAS_COMPUTE_32I.
     * Row-major contract (same trick as dot_2d_gpu): cuBLAS column-major
     * receives (op=T, W) as the leading operand and (op=N, X) as the
     * trailing one, which yields the row-major X @ W^T into Y. */
    const int alpha = 1;
    const int beta  = 0;
    cublasStatus_t stat = cublasGemmEx(
        handle,
        CUBLAS_OP_T,                /* W is [N, K]; transposed → [K, N] in col-major view */
        CUBLAS_OP_N,                /* X is [M, K], no transpose */
        N, M, K,
        &alpha,
        w_int8, CUDA_R_8I, ldw,
        X_int8_scratch_d, CUDA_R_8I, ldx,
        &beta,
        Y_int32_scratch_d, CUDA_R_32I, ldy,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr,
            "FATAL: cublasGemmEx INT8 failed status=%d  M=%d N=%d K=%d  "
            "ldw=%d ldx=%d ldy=%d  (x[%d x %d] w[%d x %d] y[%d x %d])\n",
            stat, M, N, K, ldw, ldx, ldy, x_r, x_c, w_r, w_c, y_r, y_c);
        abort();
    }

    /* 3. Dequant INT32 → BF16. */
    dequant_int32_to_bf16_cuda(Y_int32_scratch_d, scale_w, scale_X_scratch_d,
                               y_bf16, M, N);
}
#endif

static void print_gpu_memory_usage() {
    size_t free_byte;
    size_t total_byte;

    // Get the memory info
    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Error: cudaMemGetInfo failed: %s\n", 
                cudaGetErrorString(cuda_status));
        return;
    }

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;

    printf("--- GPU Memory Status ---\n");
    printf("Used:  %8.2f MB\n", used_db / (1024.0 * 1024.0));
    printf("Free:  %8.2f MB\n", free_db / (1024.0 * 1024.0));
    printf("Total: %8.2f MB\n", total_db / (1024.0 * 1024.0));
    printf("-------------------------\n");
}
#endif

#if defined (USE_ACCELERATE) || ( defined  (USE_ACCELERATE_X86) &&  !defined (USE_CUDA))
static void dot_2d_cpu(act_t *a,int a_r, int a_c, int lda, act_t *b,int b_r,int b_c,int ldb, act_t * c_out,int c_r, int c_c, int ldc, int transpose_b,int apply_attention_scaling ){

    float alpha = 1.0f;
    if (apply_attention_scaling) {
        alpha = 1.0f / sqrtf((float)a_c);  // shared inner dimension
        //printf("ALPHA = %f",alpha);
    }

    float beta = 0.0f;

    enum CBLAS_TRANSPOSE trans_b = transpose_b ? CblasTrans : CblasNoTrans;

    //int lda = a_c;                           // A: [a_r x a_c]
    //int ldb = b_c;                           // B: [b_c x b_r] if transposed
    //int ldc = transpose_b ? b_r : b_c;       // C: [a_r x output_cols]

    int M = a_r;                             // rows of A / C
    int K = a_c;                             // inner dim (shared)
    int N = transpose_b ? b_r : b_c;         // output columns

    cblas_sgemm(CblasRowMajor,  // row major order
                CblasNoTrans,   // a not transposed
                trans_b,        // flag to transpose b if set to true
                M,              // # rows of A or C
                N,              // Number of columns of B_effective / columns of C
                K,              // inner dimension, 
                alpha,          // Scalar alpha
                a,              // Pointer to matrix A
                lda,            // A stride
                b,              // Pointer to matrix B
                ldb,            // B stride
                beta,           // Scalar beta
                c_out,          // Pointer to matrix C 
                ldc             // C stride
            );
}
#endif

static void dot_2d(act_t *a,int a_r, int a_c, int lda, act_t *b,int b_r,int b_c,int ldb, act_t * c_out,int c_r, int c_c, int ldc, int transpose_b,int apply_attention_scaling ){

#ifdef USE_CUDA
    dot_2d_gpu(a,a_r, a_c, lda, b, b_r, b_c, ldb,  c_out, c_r,  c_c,  ldc,  transpose_b, apply_attention_scaling);
#elif defined (USE_ACCELERATE) || defined (USE_ACCELERATE_X86)
    dot_2d_cpu(a,a_r, a_c, lda, b, b_r, b_c, ldb,  c_out, c_r,  c_c,  ldc,  transpose_b, apply_attention_scaling);

#else    
    #error No backend (USE_CUDA or USE_ACCELERATE) is defined for dot_2d!
#endif
}


static void apply_casual_masking(act_t * a, int a_c, int size){
    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            a[i * a_c + j] = -INFINITY;
        }
    }
}

//prints start from 0 both for row and column
static void print_2d_tensor(char* name, act_t *a, int a_r_full_dim, int a_c_full_dim, int r_idx_to_print, int c_idx_to_print, int enable) {
    if (!enable) {
        return;
    }

    printf("--- Tensor: %s ---\n", name);
    printf("[");
    for (int i = 0; i < r_idx_to_print; i++) { // Loop for rows to print
        printf("  [");
        for (int j = 0; j < c_idx_to_print; j++) { // Loop for columns to print
            // Access element using the *full physical column dimension* (a_c_full_dim) as the stride
            printf(" %6.4f", (double)*(a + i * a_c_full_dim + j));
            if (j < c_idx_to_print - 1) printf(", ");
        }
        printf("]");
        if (i < r_idx_to_print - 1) printf(",\n");
        else printf("\n");
    }
    printf("]\n");
}


static void transpose_2d(act_t *a, int a_r, int a_c, act_t *b){
// TODO input check
    for (int i=0; i<a_r; i++){
        for (int j=0; j<a_c; j++){
            *(b + j*a_r +i) = *(a + i*a_c + j);
        }
    }
}

static void layernorm_2d(act_t *a, int a_r, int a_c,
                    weight_t * ln_gamma, weight_t * ln_beta,
                    act_t * out, float epsilon){

    for (int i=0; i < a_r; i++){
        act_t *row = a + i * a_c;
        float mean = mean_(row,a_c);
        float var = variance_(row,a_c,mean);
        for (int j=0; j<a_c; j++){
            *(out + i*a_c + j) = *(ln_gamma+j) * ((*(a + i*a_c + j) - mean)/sqrt(var+epsilon)) + *(ln_beta+j);
        }
    }
}

static void add_2d(act_t *a, int a_r, int a_c, act_t *b, act_t *out){
    for (int i=0; i<a_r; i++){
        for (int j=0; j<a_c; j++){
            *(out +i*a_c + j) = *(a +i*a_c + j) + *(b +i*a_c + j);
        }
    }
}

// if out is null addition is inplace
static void add_bias_2d(act_t *a, int a_r, int a_c, weight_t *b, act_t *out){
    act_t * tmp = out;
    if (out == NULL){ //inplace
        tmp = a;
    }
    for (int i=0; i<a_r; i++){
        for (int j=0; j<a_c; j++){
            *(tmp +i*a_c + j) = *(a +i*a_c + j) + *(b + j);
        }
    }
}

static float mean_(act_t *x, int len){
    float sum = 0.0;
    for (int i=0; i<len; i++){
        sum += *(x+i);
    }
    return (sum/(float)len);
}
static float variance_(act_t *x,int len, float mean){
    float sum = 0.0;
    for (int i=0; i<len; i++){
        sum += pow(((*(x+i)) - mean),2);
    }
    return (sum/(float)len);
}

static float gelu(float x){
    float term = sqrt(2.0/PI);
    return 0.5 * x * (1 + tanh (term * (x + 0.044715*pow(x,3))));
}

static void gelu_2d(act_t *a,int a_c, int a_r, act_t *out){
    act_t * tmp = out;
    if (out == NULL){ //inplace
        tmp = a;
    } 
    for (int i=0; i<a_r; i++){
        for (int j=0; j<a_c; j++){
            *(tmp +i*a_c + j) = gelu(*(a +i*a_c + j));
        }
    }
}

//// Globals /// 
struct timespec start,end;



#ifdef ENABLE_KV_CACHE

int kv_cache_enabled = 1;
#else
int kv_cache_enabled = 0;
#endif

const float eps = 0.00001;

weight_t wte[vocab_size][d_model] = {};
weight_t wte_T[d_model][vocab_size] = {};

weight_t wpe[ctx_len][d_model] = {};
act_t embeddings[ctx_len][d_model] = {}; // for now post positional embeddings. This would go into layer norm

act_t X_norm[ctx_len][d_model] = {};
act_t X_norm2[ctx_len][d_model] = {};


act_t X1[ctx_len][d_ff] = {};
act_t X1_out[ctx_len][d_ff] = {};
act_t X2[ctx_len][d_model] = {};
act_t X2_out[ctx_len][d_model] = {};
act_t Xf_out[ctx_len][d_model] = {};

/***  Attention (per-layer) ***/
weight_t (*W_q)[d_model][d_model];//
weight_t (*W_k)[d_model][d_model];
weight_t (*W_v)[d_model][d_model];
weight_t (*b_q)[d_model];
weight_t (*b_k)[d_model];
weight_t (*b_v)[d_model];

/* Output projections */
weight_t (*attn_proj_weight)[d_model][d_model];
weight_t (*attn_proj_bias)[d_model];

/* Feed Forward */
weight_t (*W1)[d_ff][d_model];
weight_t (*b1)[d_ff];
weight_t (*W2)[d_model][d_ff];
weight_t (*b2)[d_model];

/* Layer Norm */
weight_t (*layer_norm1_gamma)[d_model];
weight_t (*layer_norm1_beta)[d_model];
weight_t (*layer_norm2_gamma)[d_model];
weight_t (*layer_norm2_beta)[d_model];

#if defined(USE_INT8)
/* Host-side INT8 weight + scale buffers — staging area for the
 * quant8.bin load before cudaMemcpy to *_int8_d / *_scale_d. */
qweight_t (*W_q_int8)[d_model][d_model];
qweight_t (*W_k_int8)[d_model][d_model];
qweight_t (*W_v_int8)[d_model][d_model];
qweight_t (*attn_proj_int8)[d_model][d_model];
qweight_t (*W1_int8)[d_ff][d_model];
qweight_t (*W2_int8)[d_model][d_ff];

qscale_t (*W_q_scale)[d_model];
qscale_t (*W_k_scale)[d_model];
qscale_t (*W_v_scale)[d_model];
qscale_t (*attn_proj_scale)[d_model];
qscale_t (*W1_scale)[d_ff];
qscale_t (*W2_scale)[d_model];
#endif

#ifdef USE_CUDA
int *tokens_d;
act_t (*embeddings_d)[d_model];
act_t (*X_norm_d)[d_model];
act_t (*X_norm2_d)[d_model];
act_t (*residual_out_d)[d_model];
act_t (*residual2_out_d)[d_model];

/***  Global weights ***/
weight_t (*wte_d)[d_model];
weight_t (*wte_T_d)[vocab_size];
weight_t (*wpe_d)[d_model];

/***  Attention (per-layer) ***/
weight_t (*W_q_d)[d_model][d_model];
weight_t (*W_k_d)[d_model][d_model];
weight_t (*W_v_d)[d_model][d_model];
weight_t (*b_q_d)[d_model];
weight_t (*b_k_d)[d_model];
weight_t (*b_v_d)[d_model];

/* Output projections */
weight_t (*attn_proj_weight_d)[d_model][d_model];
weight_t (*attn_proj_bias_d)[d_model];

/* Feed Forward */
weight_t (*W1_d)[d_ff][d_model];
weight_t (*b1_d)[d_ff];
weight_t (*W2_d)[d_model][d_ff];
weight_t (*b2_d)[d_model];

/* Layer Norm */
weight_t (*layer_norm1_gamma_d)[d_model];
weight_t (*layer_norm1_beta_d)[d_model];
weight_t (*layer_norm2_gamma_d)[d_model];
weight_t (*layer_norm2_beta_d)[d_model];

weight_t *layer_normf_gamma_d;       // Final Layer Norm Gamma [d_model]
weight_t *layer_normf_beta_d;        // Final Layer Norm Beta [d_model]

#if defined(USE_INT8)
/* INT8 quantized weights — parallel to the BF16 pointers above. Only the
 * 4 large matmul tensors per layer are quantized; the preserved tensors
 * (LN, biases, embeddings) keep their weight_t storage. Each quantized
 * tensor has a matching per-output-channel FP32 scale vector.
 *
 * QKV is split at load time into Q/K/V buffers so the existing 3-GEMM
 * forward path can be reused without restructuring. */
qweight_t (*W_q_int8_d)[d_model][d_model];
qweight_t (*W_k_int8_d)[d_model][d_model];
qweight_t (*W_v_int8_d)[d_model][d_model];
qweight_t (*attn_proj_int8_d)[d_model][d_model];
qweight_t (*W1_int8_d)[d_ff][d_model];
qweight_t (*W2_int8_d)[d_model][d_ff];

qscale_t (*W_q_scale_d)[d_model];          // per-output-channel scale, length d_model per layer
qscale_t (*W_k_scale_d)[d_model];
qscale_t (*W_v_scale_d)[d_model];
qscale_t (*attn_proj_scale_d)[d_model];
qscale_t (*W1_scale_d)[d_ff];              // W1 output dim is d_ff
qscale_t (*W2_scale_d)[d_model];

/* Per-GEMM scratch buffers for the INT8 path. Sized for the largest GEMM
 * shape used in any layer (ctx_len × d_ff, set by the W1 / W2 GEMMs).
 * Reused across all GEMMs since dot_2d_gpu_int8 sequences quant → GEMM →
 * dequant within itself (stream-ordered, no cross-call aliasing). */
qweight_t *X_int8_scratch_d;     /* [ctx_len * d_ff] activation INT8 */
qscale_t  *scale_X_scratch_d;    /* [ctx_len]        per-token amax / 127 */
int32_t   *Y_int32_scratch_d;    /* [ctx_len * d_ff] raw GEMM accumulator */
#endif

act_t (*K_cache_d)[ctx_len][d_model]; // Pointer to layers of [ctx_len][d_model]
act_t (*V_cache_d)[ctx_len][d_model];

act_t (*Q_d)[d_model];

act_t (*scores_h_d)[ctx_len];
act_t (*weights_h_d)[ctx_len];

act_t (*context_d)[d_model];


act_t (*X1_d)[d_ff];
act_t (*X1_out_d)[d_ff];
act_t (*X2_d)[d_model];
act_t (*X2_out_d)[d_model];
act_t (*Xf_out_d)[d_model];

act_t (*final_attention_output_d)[d_model];
act_t (*context_heads_d)[ctx_len][head_dim];

act_t (*logits_d)[vocab_size];

#endif

weight_t temp_attn_weight[3*d_model][d_model] = {}; // 2304 = 768 * 3
weight_t temp_attn_bias[3*d_model] = {};

act_t Q[ctx_len][d_model] = {};
act_t K[ctx_len][d_model] = {};
//act_t K_cache[ctx_len][d_model] = {};
act_t K_cache[num_layers][ctx_len][d_model] = {};

act_t K_T[d_model][ctx_len] = {};
act_t V[ctx_len][d_model] = {};
//act_t V_cache[ctx_len][d_model] = {};
act_t V_cache[num_layers][ctx_len][d_model] = {};
act_t attention_scores[ctx_len][ctx_len] = {};
act_t attention_scores_temp[ctx_len][ctx_len] = {};
act_t attention_weights[ctx_len][ctx_len] = {};
act_t context[ctx_len][d_model] = {};


weight_t layer_normf_gamma[d_model] = {}; // default: no scaling
weight_t layer_normf_beta[d_model] = {};  // default: no shifting

act_t residual_out[ctx_len][d_model] = {};
act_t residual2_out[ctx_len][d_model] = {};

act_t logits[ctx_len][vocab_size] = {};

act_t context_heads[nof_heads][ctx_len][head_dim] = {};
act_t scores_h[ctx_len][ctx_len] = {};
act_t weights_h[ctx_len][ctx_len] = {};
act_t final_attention_output[ctx_len][d_model] = {};


typedef struct{
    weight_t * W_q;
    weight_t * W_k;
    weight_t * W_v;
    weight_t * b_q;
    weight_t * b_k;
    weight_t * b_v;
    weight_t *attn_proj_weight;
    weight_t *attn_proj_bias;
    weight_t * W1;
    weight_t * W2;
    weight_t * b1;
    weight_t * b2;
    weight_t *ln1_gamma;
    weight_t *ln1_beta;
    weight_t *ln2_gamma;
    weight_t *ln2_beta;

#if defined(USE_INT8)
    /* INT8 GEMM weights + per-output-channel FP32 scales. Parallel to the
     * BF16 pointers above; the BF16 ones go unused under USE_INT8 (kept to
     * avoid restructuring the struct conditionally). */
    qweight_t *W_q_int8;
    qweight_t *W_k_int8;
    qweight_t *W_v_int8;
    qweight_t *attn_proj_int8;
    qweight_t *W1_int8;
    qweight_t *W2_int8;
    qscale_t  *W_q_scale;
    qscale_t  *W_k_scale;
    qscale_t  *W_v_scale;
    qscale_t  *attn_proj_scale;
    qscale_t  *W1_scale;
    qscale_t  *W2_scale;
#endif

}TransformerBlockParams;

#ifdef USE_CUDA
static void transformer_block_gpu(act_t *input,int n_tokens,int n_new_tokens,
                        TransformerBlockParams * tbp,json_t *json_root,int layer_id,int token_idx);
#endif

static TransformerBlockParams layer[num_layers];
static void transformer_block_cpu(act_t *input,int n_tokens,int n_new_tokens,
                        TransformerBlockParams * tbp,json_t *json_root,int layer_id,int token_idx
                        );


#ifdef USE_CUDA
static void allocate_weights_gpu(void){
    CUDA_CHECK(cudaMalloc((void **)&wte_d, vocab_size * sizeof *wte_d));
    CUDA_CHECK(cudaMalloc((void **)&wpe_d, ctx_len * sizeof *wpe_d));
    CUDA_CHECK(cudaMalloc((void **)&wte_T_d, d_model * sizeof *wte_T_d));

    CUDA_CHECK(cudaMalloc((void **)&W_q_d, num_layers * sizeof *W_q_d));
    CUDA_CHECK(cudaMalloc((void **)&W_k_d, num_layers * sizeof *W_k_d));
    CUDA_CHECK(cudaMalloc((void **)&W_v_d, num_layers * sizeof *W_v_d));
    CUDA_CHECK(cudaMalloc((void **)&b_q_d, num_layers * sizeof *b_q_d));
    CUDA_CHECK(cudaMalloc((void **)&b_k_d, num_layers * sizeof *b_k_d));
    CUDA_CHECK(cudaMalloc((void **)&b_v_d, num_layers * sizeof *b_v_d));
    CUDA_CHECK(cudaMalloc((void **)&attn_proj_weight_d, num_layers * sizeof *attn_proj_weight_d));
    CUDA_CHECK(cudaMalloc((void **)&attn_proj_bias_d, num_layers * sizeof *attn_proj_bias_d));
    CUDA_CHECK(cudaMalloc((void **)&W1_d, num_layers * sizeof *W1_d));
    CUDA_CHECK(cudaMalloc((void **)&W2_d, num_layers * sizeof *W2_d));
    CUDA_CHECK(cudaMalloc((void **)&b1_d, num_layers * sizeof *b1_d));
    CUDA_CHECK(cudaMalloc((void **)&b2_d, num_layers * sizeof *b2_d));
    CUDA_CHECK(cudaMalloc((void **)&layer_norm1_gamma_d, num_layers * sizeof *layer_norm1_gamma_d));
    CUDA_CHECK(cudaMalloc((void **)&layer_norm1_beta_d, num_layers * sizeof *layer_norm1_beta_d));
    CUDA_CHECK(cudaMalloc((void **)&layer_norm2_gamma_d, num_layers * sizeof *layer_norm2_gamma_d));
    CUDA_CHECK(cudaMalloc((void **)&layer_norm2_beta_d, num_layers * sizeof *layer_norm2_beta_d));

    CUDA_CHECK(cudaMalloc((void **)&layer_normf_gamma_d, d_model * sizeof(weight_t)));
    CUDA_CHECK(cudaMalloc((void **)&layer_normf_beta_d, d_model * sizeof(weight_t)));

#if defined(USE_INT8)
    /* INT8 weight + per-output-channel FP32 scale buffers. The 4 large matmul
     * tensors per layer (W_qkv split into Q/K/V, attn_proj, W1, W2). The BF16
     * counterparts above remain allocated but go unused under USE_INT8 — kept
     * to avoid a structural fork of the alloc/free path in this phase. */
    CUDA_CHECK(cudaMalloc((void **)&W_q_int8_d,        num_layers * sizeof *W_q_int8_d));
    CUDA_CHECK(cudaMalloc((void **)&W_k_int8_d,        num_layers * sizeof *W_k_int8_d));
    CUDA_CHECK(cudaMalloc((void **)&W_v_int8_d,        num_layers * sizeof *W_v_int8_d));
    CUDA_CHECK(cudaMalloc((void **)&attn_proj_int8_d,  num_layers * sizeof *attn_proj_int8_d));
    CUDA_CHECK(cudaMalloc((void **)&W1_int8_d,         num_layers * sizeof *W1_int8_d));
    CUDA_CHECK(cudaMalloc((void **)&W2_int8_d,         num_layers * sizeof *W2_int8_d));

    CUDA_CHECK(cudaMalloc((void **)&W_q_scale_d,        num_layers * sizeof *W_q_scale_d));
    CUDA_CHECK(cudaMalloc((void **)&W_k_scale_d,        num_layers * sizeof *W_k_scale_d));
    CUDA_CHECK(cudaMalloc((void **)&W_v_scale_d,        num_layers * sizeof *W_v_scale_d));
    CUDA_CHECK(cudaMalloc((void **)&attn_proj_scale_d,  num_layers * sizeof *attn_proj_scale_d));
    CUDA_CHECK(cudaMalloc((void **)&W1_scale_d,         num_layers * sizeof *W1_scale_d));
    CUDA_CHECK(cudaMalloc((void **)&W2_scale_d,         num_layers * sizeof *W2_scale_d));

    /* Per-GEMM scratch — sized for the largest shape (ctx_len × d_ff). */
    CUDA_CHECK(cudaMalloc((void **)&X_int8_scratch_d,  (size_t)ctx_len * d_ff * sizeof(qweight_t)));
    CUDA_CHECK(cudaMalloc((void **)&scale_X_scratch_d, (size_t)ctx_len           * sizeof(qscale_t)));
    CUDA_CHECK(cudaMalloc((void **)&Y_int32_scratch_d, (size_t)ctx_len * d_ff * sizeof(int32_t)));
#endif

    CUDA_CHECK(cudaMalloc((void **)&tokens_d, ctx_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&embeddings_d, ctx_len * sizeof *embeddings_d));

    CUDA_CHECK(cudaMalloc((void **)&X_norm_d, ctx_len * sizeof *X_norm_d));
    CUDA_CHECK(cudaMalloc((void **)&X_norm2_d, ctx_len * sizeof *X_norm2_d));
    CUDA_CHECK(cudaMalloc((void **)&residual_out_d, ctx_len * sizeof *residual_out_d));
    CUDA_CHECK(cudaMalloc((void **)&residual2_out_d, ctx_len * sizeof *residual2_out_d));

    CUDA_CHECK(cudaMalloc((void **)&Q_d, ctx_len * sizeof *Q_d));
    CUDA_CHECK(cudaMalloc((void **)&K_cache_d, num_layers * sizeof *K_cache_d));
    CUDA_CHECK(cudaMalloc((void **)&V_cache_d, num_layers * sizeof *V_cache_d));

    CUDA_CHECK(cudaMalloc((void **)&scores_h_d, ctx_len * sizeof *scores_h_d));
    CUDA_CHECK(cudaMalloc((void **)&weights_h_d, ctx_len * sizeof *weights_h_d));
    CUDA_CHECK(cudaMalloc((void **)&context_d, ctx_len * sizeof *context_d));

    CUDA_CHECK(cudaMalloc((void **)&X1_d, ctx_len * sizeof *X1_d));
    CUDA_CHECK(cudaMalloc((void **)&X1_out_d, ctx_len * sizeof *X1_out_d));
    CUDA_CHECK(cudaMalloc((void **)&X2_d, ctx_len * sizeof *X2_d));
    CUDA_CHECK(cudaMalloc((void **)&X2_out_d, ctx_len * sizeof *X2_out_d));
    CUDA_CHECK(cudaMalloc((void **)&Xf_out_d, ctx_len * sizeof *Xf_out_d));

    CUDA_CHECK(cudaMalloc((void **)&final_attention_output_d, ctx_len * sizeof *final_attention_output_d));

    CUDA_CHECK(cudaMalloc((void **)&context_heads_d, nof_heads * sizeof *context_heads_d));

    CUDA_CHECK(cudaMalloc((void **)&logits_d, ctx_len * sizeof *logits_d));

}

#endif
static void allocate_weights_cpu(void){
    size_t allocated_size = 0;

    /* Under USE_INT8 the 4 large matmul weights per layer come from the
     * quant8.bin as INT8 + scale and live in the *_int8 / *_scale host
     * staging buffers below. The BF16 W_q/W_k/W_v/attn_proj_weight/W1/W2
     * host buffers are unused in that build, so skip their mallocs.
     * Biases and LN params are preserved tensors — still allocated. */
#if !defined(USE_INT8)
    allocated_size += num_layers * sizeof *W_q;
    W_q   =  malloc(num_layers * sizeof *W_q);

    allocated_size += num_layers * sizeof *W_k;
    W_k   =  malloc(num_layers * sizeof *W_k);

    allocated_size += num_layers * sizeof *W_v;
    W_v   =  malloc(num_layers * sizeof *W_v);
#endif

    allocated_size += num_layers * sizeof *b_q;
    b_q   =  malloc(num_layers * sizeof *b_q);

    allocated_size += num_layers * sizeof *b_k;
    b_k   =  malloc(num_layers * sizeof *b_k);

    allocated_size += num_layers * sizeof *b_v;
    b_v   =  malloc(num_layers * sizeof *b_v);

#if !defined(USE_INT8)
    allocated_size += num_layers * sizeof *attn_proj_weight;
    attn_proj_weight = malloc(num_layers * sizeof *attn_proj_weight);
#endif

    allocated_size += num_layers * sizeof *attn_proj_bias;
    attn_proj_bias   = malloc(num_layers * sizeof *attn_proj_bias);

#if !defined(USE_INT8)
    allocated_size += num_layers * sizeof *W1;
    W1  = malloc(num_layers * sizeof *W1);

    allocated_size += num_layers * sizeof *W2;
    W2  = malloc(num_layers * sizeof *W2);
#endif

    allocated_size += num_layers * sizeof *b1;
    b1  = malloc(num_layers * sizeof *b1);

    allocated_size += num_layers * sizeof *b2;
    b2  = malloc(num_layers * sizeof *b2);

    allocated_size += num_layers * sizeof *layer_norm1_gamma;
    layer_norm1_gamma = malloc(num_layers * sizeof *layer_norm1_gamma);

    allocated_size += num_layers * sizeof *layer_norm1_beta;
    layer_norm1_beta  = malloc(num_layers * sizeof *layer_norm1_beta);

    allocated_size += num_layers * sizeof *layer_norm2_gamma;
    layer_norm2_gamma = malloc(num_layers * sizeof *layer_norm2_gamma);

    allocated_size += num_layers * sizeof *layer_norm2_beta;
    layer_norm2_beta  = malloc(num_layers * sizeof *layer_norm2_beta);


    // Small 340M, Medium 1.2G, Large 2.8G
    //printf("Total allocated memory = %zu\n",allocated_size);
                                
#if !defined(USE_INT8)
    if (!W_q || !W_k) { perror("malloc"); exit(1); }
#endif

#if defined(USE_INT8)
    /* Host-side INT8 staging buffers. Same per-layer flat layout as the
     * BF16 buffers above; populated by load_all_weights from quant8.bin
     * and copied to *_int8_d / *_scale_d in copy_weights_to_gpu. */
    W_q_int8        = malloc(num_layers * sizeof *W_q_int8);
    W_k_int8        = malloc(num_layers * sizeof *W_k_int8);
    W_v_int8        = malloc(num_layers * sizeof *W_v_int8);
    attn_proj_int8  = malloc(num_layers * sizeof *attn_proj_int8);
    W1_int8         = malloc(num_layers * sizeof *W1_int8);
    W2_int8         = malloc(num_layers * sizeof *W2_int8);

    W_q_scale        = malloc(num_layers * sizeof *W_q_scale);
    W_k_scale        = malloc(num_layers * sizeof *W_k_scale);
    W_v_scale        = malloc(num_layers * sizeof *W_v_scale);
    attn_proj_scale  = malloc(num_layers * sizeof *attn_proj_scale);
    W1_scale         = malloc(num_layers * sizeof *W1_scale);
    W2_scale         = malloc(num_layers * sizeof *W2_scale);

    if (!W_q_int8 || !W_q_scale || !W1_int8 || !W2_int8) {
        perror("malloc (int8 staging)"); exit(1);
    }
#endif

}

static void update_layer_table(void){
#ifdef USE_CUDA
    for (int l=0; l < num_layers ; l++){
        layer[l].W_q = &W_q_d[l][0][0];
        layer[l].W_k = &W_k_d[l][0][0];
        layer[l].W_v = &W_v_d[l][0][0];
        layer[l].b_k = &b_k_d[l][0];
        layer[l].b_q = &b_q_d[l][0];
        layer[l].b_v = &b_v_d[l][0];
        layer[l].attn_proj_weight = &attn_proj_weight_d[l][0][0];
        layer[l].attn_proj_bias = &attn_proj_bias_d[l][0];
        layer[l].W1 = &W1_d[l][0][0];
        layer[l].W2 = &W2_d[l][0][0];
        layer[l].b1 = &b1_d[l][0];
        layer[l].b2 = &b2_d[l][0];
        layer[l].ln1_gamma = &layer_norm1_gamma_d[l][0];
        layer[l].ln1_beta = &layer_norm1_beta_d[l][0];
        layer[l].ln2_gamma = &layer_norm2_gamma_d[l][0];
        layer[l].ln2_beta = &layer_norm2_beta_d[l][0];

#if defined(USE_INT8)
        layer[l].W_q_int8       = &W_q_int8_d[l][0][0];
        layer[l].W_k_int8       = &W_k_int8_d[l][0][0];
        layer[l].W_v_int8       = &W_v_int8_d[l][0][0];
        layer[l].attn_proj_int8 = &attn_proj_int8_d[l][0][0];
        layer[l].W1_int8        = &W1_int8_d[l][0][0];
        layer[l].W2_int8        = &W2_int8_d[l][0][0];
        layer[l].W_q_scale       = &W_q_scale_d[l][0];
        layer[l].W_k_scale       = &W_k_scale_d[l][0];
        layer[l].W_v_scale       = &W_v_scale_d[l][0];
        layer[l].attn_proj_scale = &attn_proj_scale_d[l][0];
        layer[l].W1_scale        = &W1_scale_d[l][0];
        layer[l].W2_scale        = &W2_scale_d[l][0];
#endif
    }
#endif
}
static void allocate_weights(void){
#ifdef USE_CUDA
    allocate_weights_gpu();
    // easiest for now,used to copy weights to GPU memory
    allocate_weights_cpu();
#else
    allocate_weights_cpu();
#endif
}

static void init_layer_table(void){
    for (int l=0; l < num_layers ; l++){
        layer[l].W_q = &W_q[l][0][0];
        layer[l].W_k = &W_k[l][0][0];
        layer[l].W_v = &W_v[l][0][0];
        layer[l].b_k = &b_k[l][0];
        layer[l].b_q = &b_q[l][0];
        layer[l].b_v = &b_v[l][0];
        layer[l].attn_proj_weight = &attn_proj_weight[l][0][0];
        layer[l].attn_proj_bias = &attn_proj_bias[l][0];
        layer[l].W1 = &W1[l][0][0];
        layer[l].W2 = &W2[l][0][0];
        layer[l].b1 = &b1[l][0];
        layer[l].b2 = &b2[l][0];
        layer[l].ln1_gamma = &layer_norm1_gamma[l][0];
        layer[l].ln1_beta = &layer_norm1_beta[l][0];
        layer[l].ln2_gamma = &layer_norm2_gamma[l][0];
        layer[l].ln2_beta = &layer_norm2_beta[l][0];
    }
}

int glob_idx = 0;
int last_index = 0; // for kv cache


static void transformer_block(act_t *input,int n_tokens,int n_new_tokens,
                        TransformerBlockParams * tbp,json_t *json_root,int layer_id,int token_idx
                        )
{
#ifdef USE_CUDA
    transformer_block_gpu(input,n_tokens,n_new_tokens,tbp,json_root,layer_id,token_idx);
#else
    transformer_block_cpu(input,n_tokens,n_new_tokens,tbp,json_root,layer_id,token_idx);
#endif
}

#ifdef USE_CUDA
static void transformer_block_gpu(act_t *input,int n_tokens,int n_new_tokens,
                        TransformerBlockParams * tbp,json_t *json_root,int layer_id,int token_idx
                        ){
    //char layer_key[32];
    //snprintf(layer_key, sizeof(layer_key), "token_%d_layer_%d", token_idx,layer_id);
    //json_t *layer_obj = json_object();
    
    // Layer Norm 1
    //printf("--- Layer (Detailed Log) ---\n");
    //printf("Input:\n");
    int cache_start_index = n_tokens - n_new_tokens;

    layernorm_cuda((act_t (*)[d_model])(input + cache_start_index*d_model),n_new_tokens,d_model,tbp->ln1_gamma,tbp->ln1_beta, (act_t (*)[d_model])&X_norm_d[cache_start_index][0],eps);

    //print_2d_tensor("LN1 X_norm[1][:10]:",&X_norm[1][0],ctx_len,d_model,1,10,0);

    //int n_new_tokens = n_tokens-last_index;

    // QKV
            
    
#if defined(USE_INT8)
    dot_2d_gpu_int8(&X_norm_d[cache_start_index][0],n_new_tokens,d_model,d_model,tbp->W_q_int8,d_model,d_model,d_model,tbp->W_q_scale,&Q_d[cache_start_index][0],n_new_tokens,d_model,d_model);
#else
    dot_2d(&X_norm_d[cache_start_index][0],n_new_tokens,d_model,d_model,tbp->W_q,d_model,d_model,d_model,&Q_d[cache_start_index][0],n_new_tokens, d_model,d_model,1,!APPLY_ATTENTION_SCALING);
#endif
    add_bias_cuda(&Q_d[cache_start_index][0],n_new_tokens,d_model,tbp->b_q,NULL);
    // final destination pointer in the cache

    act_t* k_cache_ptr = &K_cache_d[layer_id][cache_start_index][0];
#if defined(USE_INT8)
    dot_2d_gpu_int8(&X_norm_d[cache_start_index][0],n_new_tokens,d_model,d_model,tbp->W_k_int8,d_model,d_model,d_model,tbp->W_k_scale,k_cache_ptr,n_new_tokens,d_model,d_model);
#else
    dot_2d(&X_norm_d[cache_start_index][0],n_new_tokens,d_model,d_model,tbp->W_k,d_model,d_model,d_model,k_cache_ptr,n_new_tokens,d_model,d_model,1,!APPLY_ATTENTION_SCALING);
#endif
    add_bias_cuda(k_cache_ptr,n_new_tokens,d_model,tbp->b_k,NULL);

    act_t* v_cache_ptr = &V_cache_d[layer_id][cache_start_index][0];
#if defined(USE_INT8)
    dot_2d_gpu_int8(&X_norm_d[cache_start_index][0],n_new_tokens,d_model,d_model,tbp->W_v_int8,d_model,d_model,d_model,tbp->W_v_scale,v_cache_ptr,n_new_tokens,d_model,d_model);
#else
    dot_2d(&X_norm_d[cache_start_index][0],n_new_tokens,d_model,d_model,tbp->W_v,d_model,d_model,d_model,v_cache_ptr,n_new_tokens,d_model,d_model,1,!APPLY_ATTENTION_SCALING);
#endif
    add_bias_cuda(v_cache_ptr,n_new_tokens,d_model,tbp->b_v,NULL);
    
    last_index = n_tokens;
 
    
    //add_tensor_to_layer(layer_obj, "V", &V[0][0], ctx_len, d_model, n_tokens, d_model);
    //print_2d_tensor("C Q[1][:10]:",&Q[1][0],ctx_len,d_model,1,10,0);
    //print_2d_tensor("C K[1][:10]:",&K[1][0],ctx_len,d_model,1,10,0);
    //print_2d_tensor("C V[1][:10]:",&V[1][0],ctx_len,d_model,1,10,0);


    // ** to add here the multi head attention code ***
    /// To optimize this loop. No need to recompute entire attention matrix at every single
    // step. Need to calc attention for the last token only during the generation 
    // phase. which means calc 1xn_tokens instead of n_tokens x n_tokens matrix
    for (int h=0 ; h < nof_heads; h++){
        act_t *q_h;
        act_t *k_h;
        act_t *v_h;
        act_t *context_h_out = &context_heads_d[h][0][0];

        k_h = &K_cache_d[layer_id][0][0]+ h * head_dim;
        v_h = &V_cache_d[layer_id][0][0]+ h * head_dim;

        if(n_new_tokens == 1){
            act_t* q_last_token_h = &Q_d[n_tokens - 1][0] + h * head_dim;
            act_t* scores_last_row = &scores_h_d[n_tokens - 1][0];
            act_t* weights_last_row = &weights_h_d[n_tokens - 1][0];
            act_t* context_last_row = context_h_out + (n_tokens - 1) * head_dim;
             // 1. Calculate scores: Q_last dot K_all -> [1 x head_dim] @ [head_dim x n_tokens] = [1 x n_tokens]
            dot_2d(q_last_token_h, 1, head_dim, d_model, k_h, n_tokens, head_dim, d_model, scores_last_row, 1, n_tokens, ctx_len, 1, APPLY_ATTENTION_SCALING);
            
            // Causal mask is implicit up to n_tokens-1, no need to apply for the last row.
            // 2. Softmax on the single row of scores (no causal mask: the single row attends to all prior tokens)
            softmax_cuda(scores_last_row, 1, n_tokens, ctx_len, weights_last_row, 1.0f, /*causal_mask=*/0);
            // 3. Calculate context: Weights_last dot V_all -> [1 x n_tokens] @ [n_tokens x head_dim] = [1 x head_dim]
            dot_2d(weights_last_row, 1, n_tokens, ctx_len, v_h, n_tokens, head_dim, d_model, context_last_row, 1, head_dim, head_dim, 0, !APPLY_ATTENTION_SCALING);
        } else {
            // prefill
            q_h = &Q_d[0][0]+ h * head_dim;
            dot_2d(q_h,n_tokens,head_dim,d_model,k_h,n_tokens,head_dim,d_model,&scores_h_d[0][0],n_tokens,n_tokens,ctx_len,1,APPLY_ATTENTION_SCALING);
            // Causal masking is now folded into softmax_cuda below (see cuda/softmax.cu).
            // The standalone kernel was dropped — it was the dominant BF16 prefill penalty
            // (+100 ms on Large at 1024 tokens), and it ran 720× per token (per layer × per head)
            // for a pattern that the softmax kernel can express with a single column-index check.
            // casual_masking_cuda(&scores_h_d[0][0],ctx_len/*n_tokens*/,n_tokens);
            //print_2d_tensor("C scores_h[1][i] (before Softmax):",&scores_h[1][0],ctx_len,ctx_len,1,10,0);
            softmax_cuda(&scores_h_d[0][0], n_tokens,n_tokens,ctx_len, &weights_h_d[0][0], 1.0f, /*causal_mask=*/1);
            //print_2d_tensor("C weights_h[1][:10]",&weights_h[1][0],ctx_len,ctx_len,1,10,0);
            dot_2d(&weights_h_d[0][0],n_tokens,n_tokens,ctx_len,v_h,n_tokens,head_dim,d_model,context_h_out,n_tokens,head_dim,head_dim,0,!APPLY_ATTENTION_SCALING);
        }
                               
    }

    if(n_new_tokens == 1){
        // generation phase

        // index of the new token
        int i = n_tokens - 1; // also cache_start_index

        // 1. head concat (for the last token only)
        //for (int h = 0; h < nof_heads; h++) {
            // More efficient to copy the whole head's output for the token at once
            // ? ? ?
        concat_heads_cuda(&context_heads_d[0][0][0],&final_attention_output_d[0][0],i,nof_heads,head_dim,ctx_len);
            //memcpy(&final_attention_output[i][h * head_dim], &context_heads[h][i][0], head_dim * sizeof(float));
        //}
        // 2. Attention projection (on the last token only)
#if defined(USE_INT8)
        dot_2d_gpu_int8(&final_attention_output_d[i][0],1,d_model,d_model,tbp->attn_proj_int8,d_model,d_model,d_model,tbp->attn_proj_scale,&context_d[i][0],1,d_model,d_model);
#else
        dot_2d(&final_attention_output_d[i][0],1,d_model,d_model,tbp->attn_proj_weight,d_model,d_model,d_model,&context_d[i][0],1,d_model,d_model,1,!APPLY_ATTENTION_SCALING);
#endif

        // Attn projection bias
        add_bias_cuda(&context_d[i][0],1,d_model,tbp->attn_proj_bias,NULL);

        // 3. Residual connection
        add_2d_cuda(input + (i * d_model),1,d_model,&context_d[i][0],&residual_out_d[i][0]);


        // 4. Layer Norm 2 (on the last token only)
        layernorm_cuda((act_t (*)[d_model])&residual_out_d[i][0],1,d_model,tbp->ln2_gamma,tbp->ln2_beta, (act_t (*)[d_model])&X_norm2_d[i][0],eps);


        // 5. MLP (on the last token only)
#if defined(USE_INT8)
        dot_2d_gpu_int8(&X_norm2_d[i][0],1,d_model,d_model,tbp->W1_int8,d_ff,d_model,d_model,tbp->W1_scale,&X1_out_d[i][0],1,d_ff,d_ff);
#else
        dot_2d(&X_norm2_d[i][0],1,d_model,d_model,tbp->W1,d_ff,d_model,d_model,&X1_out_d[i][0],1,d_ff,d_ff,1,!APPLY_ATTENTION_SCALING);
#endif
        // W1 bias
        add_bias_cuda(&X1_out_d[i][0],1,d_ff,tbp->b1,NULL);
        // GELU activation
        gelu_cuda(&X1_out_d[i][0],d_ff,1,NULL);
        // W2
#if defined(USE_INT8)
        dot_2d_gpu_int8(&X1_out_d[i][0],1,d_ff,d_ff,tbp->W2_int8,d_model,d_ff,d_ff,tbp->W2_scale,&X2_out_d[i][0],1,d_model,d_model);
#else
        dot_2d(&X1_out_d[i][0],1,d_ff,d_ff,tbp->W2,d_model,d_ff,d_ff,&X2_out_d[i][0],1,d_model,d_model,1,!APPLY_ATTENTION_SCALING);
#endif
        // W2 bias
        add_bias_cuda(&X2_out_d[i][0],1,d_model,tbp->b2,NULL);

        // 6. Final Residual Connection (for the last token only)
        // First, preserve the state of previous tokens by copying them over
        if (i > 0) {
            cudaMemcpy(&residual2_out_d[0][0], input, i * d_model * sizeof(act_t), cudaMemcpyDeviceToDevice);
            //memcpy(&residual2_out[0][0], input, i * d_model * sizeof(float));
        }
        // Then, calculate the new residual for the last token /////CUDA is missing///
        //add_2d(&X2_out[0][0],n_tokens,d_model,&residual_out[0][0],&residual2_out[0][0]);
        add_2d_cuda(&residual_out_d[i][0], 1, d_model, &X2_out_d[i][0], &residual2_out_d[i][0]);

    } else {
        // prefill phase — run the post-attention pipeline for all n_tokens rows

        // 1. head concat for ALL tokens
        // TODO: replace with a single (token x head x dim) grid kernel for perf
        for (int t = 0; t < n_tokens; t++) {
            concat_heads_cuda(&context_heads_d[0][0][0],&final_attention_output_d[0][0],t,nof_heads,head_dim,ctx_len);
        }

        // 2. Attention projection: [n_tokens x d_model] @ W_proj^T -> context
#if defined(USE_INT8)
        dot_2d_gpu_int8(&final_attention_output_d[0][0],n_tokens,d_model,d_model,tbp->attn_proj_int8,d_model,d_model,d_model,tbp->attn_proj_scale,&context_d[0][0],n_tokens,d_model,d_model);
#else
        dot_2d(&final_attention_output_d[0][0],n_tokens,d_model,d_model,tbp->attn_proj_weight,d_model,d_model,d_model,&context_d[0][0],n_tokens,d_model,d_model,1,!APPLY_ATTENTION_SCALING);
#endif
        // Attn projection bias
        add_bias_cuda(&context_d[0][0],n_tokens,d_model,tbp->attn_proj_bias,NULL);

        // 3. Residual connection
        add_2d_cuda(input,n_tokens,d_model,&context_d[0][0],&residual_out_d[0][0]);

        // 4. Layer Norm 2
        layernorm_cuda((act_t (*)[d_model])&residual_out_d[0][0],n_tokens,d_model,tbp->ln2_gamma,tbp->ln2_beta,(act_t (*)[d_model])&X_norm2_d[0][0],eps);

        // 5. MLP
#if defined(USE_INT8)
        dot_2d_gpu_int8(&X_norm2_d[0][0],n_tokens,d_model,d_model,tbp->W1_int8,d_ff,d_model,d_model,tbp->W1_scale,&X1_out_d[0][0],n_tokens,d_ff,d_ff);
#else
        dot_2d(&X_norm2_d[0][0],n_tokens,d_model,d_model,tbp->W1,d_ff,d_model,d_model,&X1_out_d[0][0],n_tokens,d_ff,d_ff,1,!APPLY_ATTENTION_SCALING);
#endif
        // W1 bias
        add_bias_cuda(&X1_out_d[0][0],n_tokens,d_ff,tbp->b1,NULL);
        // GELU activation
        gelu_cuda(&X1_out_d[0][0],d_ff,n_tokens,NULL);
        // W2
#if defined(USE_INT8)
        dot_2d_gpu_int8(&X1_out_d[0][0],n_tokens,d_ff,d_ff,tbp->W2_int8,d_model,d_ff,d_ff,tbp->W2_scale,&X2_out_d[0][0],n_tokens,d_model,d_model);
#else
        dot_2d(&X1_out_d[0][0],n_tokens,d_ff,d_ff,tbp->W2,d_model,d_ff,d_ff,&X2_out_d[0][0],n_tokens,d_model,d_model,1,!APPLY_ATTENTION_SCALING);
#endif
        // W2 bias
        add_bias_cuda(&X2_out_d[0][0],n_tokens,d_model,tbp->b2,NULL);

        // 6. Final Residual Connection (for all tokens)
        add_2d_cuda(&residual_out_d[0][0],n_tokens,d_model,&X2_out_d[0][0],&residual2_out_d[0][0]);
    }
    
    //json_object_set_new(json_root, layer_key, layer_obj);
                        
}
#endif


static void transformer_block_cpu(act_t *input,int n_tokens,int n_new_tokens,
                        TransformerBlockParams * tbp,json_t *json_root,int layer_id,int token_idx
                        ){
    
    char layer_key[32];
    snprintf(layer_key, sizeof(layer_key), "token_%d_layer_%d", token_idx,layer_id);
    json_t *layer_obj = json_object();
    
    // Layer Norm 1
    //printf("--- Layer (Detailed Log) ---\n");
    //printf("Input:\n");

    layernorm_2d(input,n_tokens,d_model,tbp->ln1_gamma,tbp->ln1_beta, &X_norm[0][0],eps);
    print_2d_tensor("LN1 X_norm[1][:10]:",&X_norm[1][0],ctx_len,d_model,1,10,0);

    //int n_new_tokens = n_tokens-last_index;

    // QKV
    if(kv_cache_enabled){
        
        int cache_start_index = n_tokens - n_new_tokens;

        dot_2d(&X_norm[cache_start_index][0],n_new_tokens,d_model,d_model,tbp->W_q,d_model,d_model,d_model,&Q[cache_start_index][0],n_new_tokens, d_model,d_model,1,!APPLY_ATTENTION_SCALING);
        add_bias_2d(&Q[cache_start_index][0],n_new_tokens,d_model,tbp->b_q,NULL);

        // final destination pointer in the cache
        act_t* k_cache_ptr = &K_cache[layer_id][cache_start_index][0];
        dot_2d(&X_norm[cache_start_index][0],n_new_tokens,d_model,d_model,tbp->W_k,d_model,d_model,d_model,k_cache_ptr,n_new_tokens,d_model,d_model,1,!APPLY_ATTENTION_SCALING);
        add_bias_2d(k_cache_ptr,n_new_tokens,d_model,tbp->b_k,NULL);
        //memcpy(&K_cache[layer_id][cache_start_index][0],&K[0][0], n_new_tokens * d_model * sizeof(float));

        act_t* v_cache_ptr = &V_cache[layer_id][cache_start_index][0];
        dot_2d(&X_norm[cache_start_index][0],n_new_tokens,d_model,d_model,tbp->W_v,d_model,d_model,d_model,v_cache_ptr,n_new_tokens,d_model,d_model,1,!APPLY_ATTENTION_SCALING);
        add_bias_2d(v_cache_ptr,n_new_tokens,d_model,tbp->b_v,NULL);
        //memcpy(&V_cache[layer_id][cache_start_index][0],&V[0][0], n_new_tokens * d_model * sizeof(float));

        last_index = n_tokens;

    } else {
        dot_2d(&X_norm[0][0],n_tokens,d_model,d_model,tbp->W_q,d_model,d_model,d_model,&Q[0][0],n_tokens, d_model,d_model,1,!APPLY_ATTENTION_SCALING);
        add_bias_2d(&Q[0][0],n_tokens,d_model,tbp->b_q,NULL);

        dot_2d(&X_norm[0][0],n_tokens,d_model,d_model,tbp->W_k,d_model,d_model,d_model,&K[0][0],n_tokens,d_model,d_model,1,!APPLY_ATTENTION_SCALING);
        add_bias_2d(&K[0][0],n_tokens,d_model,tbp->b_k,NULL);

        dot_2d(&X_norm[0][0],n_tokens,d_model,d_model,tbp->W_v,d_model,d_model,d_model,&V[0][0],n_tokens,d_model,d_model,1,!APPLY_ATTENTION_SCALING);
        add_bias_2d(&V[0][0],n_tokens,d_model,tbp->b_v,NULL);
    }
    
    //add_tensor_to_layer(layer_obj, "V", &V[0][0], ctx_len, d_model, n_tokens, d_model);
    print_2d_tensor("C Q[1][:10]:",&Q[1][0],ctx_len,d_model,1,10,0);
    print_2d_tensor("C K[1][:10]:",&K[1][0],ctx_len,d_model,1,10,0);
    print_2d_tensor("C V[1][:10]:",&V[1][0],ctx_len,d_model,1,10,0);


    // ** to add here the multi head attention code ***
    /// To optimize this loop. No need to recompute entire attention matrix at every single
    // step. Need to calc attention for the last token only during the generation 
    // phase. which means calc 1xn_tokens instead of n_tokens x n_tokens matrix
    for (int h=0 ; h < nof_heads; h++){
        act_t *q_h;
        act_t *k_h;
        act_t *v_h;
        act_t *context_h_out = &context_heads[h][0][0];

        if (kv_cache_enabled){
            k_h = &K_cache[layer_id][0][0]+ h * head_dim;
            v_h = &V_cache[layer_id][0][0]+ h * head_dim;
            //context_h_out = &context_heads[h][0][0];

            if(n_new_tokens == 1){
                act_t* q_last_token_h = &Q[n_tokens - 1][0] + h * head_dim;
                act_t* scores_last_row = &scores_h[n_tokens - 1][0];
                act_t* weights_last_row = &weights_h[n_tokens - 1][0];
                act_t* context_last_row = context_h_out + (n_tokens - 1) * head_dim;

                 // 1. Calculate scores: Q_last dot K_all -> [1 x head_dim] @ [head_dim x n_tokens] = [1 x n_tokens]
                dot_2d(q_last_token_h, 1, head_dim, d_model, k_h, n_tokens, head_dim, d_model, scores_last_row, 1, n_tokens, ctx_len, 1, APPLY_ATTENTION_SCALING);
                
                // Causal mask is implicit up to n_tokens-1, no need to apply for the last row.
                // 2. Softmax on the single row of scores
                softmax_2d(scores_last_row, 1, n_tokens, ctx_len, weights_last_row);

                // 3. Calculate context: Weights_last dot V_all -> [1 x n_tokens] @ [n_tokens x head_dim] = [1 x head_dim]
                dot_2d(weights_last_row, 1, n_tokens, ctx_len, v_h, n_tokens, head_dim, d_model, context_last_row, 1, head_dim, head_dim, 0, !APPLY_ATTENTION_SCALING);


            } else {
                // prefill
                q_h = &Q[0][0]+ h * head_dim;
                dot_2d(q_h,n_tokens,head_dim,d_model,k_h,n_tokens,head_dim,d_model,&scores_h[0][0],n_tokens,n_tokens,ctx_len,1,APPLY_ATTENTION_SCALING);
                apply_casual_masking(&scores_h[0][0],ctx_len/*n_tokens*/,n_tokens);
                //print_2d_tensor("C scores_h[1][i] (before Softmax):",&scores_h[1][0],ctx_len,ctx_len,1,10,0);            
                softmax_2d(&scores_h[0][0], n_tokens,n_tokens,ctx_len, &weights_h[0][0]);        
                //print_2d_tensor("C weights_h[1][:10]",&weights_h[1][0],ctx_len,ctx_len,1,10,0);
                dot_2d(&weights_h[0][0],n_tokens,n_tokens,ctx_len,v_h,n_tokens,head_dim,d_model,context_h_out,n_tokens,head_dim,head_dim,0,!APPLY_ATTENTION_SCALING);
            }


        }else{
            q_h = &Q[0][0]+ h * head_dim;
            k_h = &K[0][0]+ h * head_dim;
            v_h = &V[0][0]+ h * head_dim;
            //context_h_out = &context_heads[h][0][0];

            //Clear scores buffer to avoid stale values
            memset(scores_h, 0, sizeof(act_t) * ctx_len * ctx_len);

            dot_2d(q_h,n_tokens,head_dim,d_model,k_h,n_tokens,head_dim,d_model,&scores_h[0][0],n_tokens,n_tokens,ctx_len,1,APPLY_ATTENTION_SCALING);
            apply_casual_masking(&scores_h[0][0],ctx_len/*n_tokens*/,n_tokens);
            //print_2d_tensor("C scores_h[1][i] (before Softmax):",&scores_h[1][0],ctx_len,ctx_len,1,10,0);            
            softmax_2d(&scores_h[0][0], n_tokens,n_tokens,ctx_len, &weights_h[0][0]);        
            //print_2d_tensor("C weights_h[1][:10]",&weights_h[1][0],ctx_len,ctx_len,1,10,0);
            dot_2d(&weights_h[0][0],n_tokens,n_tokens,ctx_len,v_h,n_tokens,head_dim,d_model,context_h_out,n_tokens,head_dim,head_dim,0,!APPLY_ATTENTION_SCALING);
        }                        
    }

    if(kv_cache_enabled && n_new_tokens == 1){
        // generation phase
            // index of the new token
            int i = n_tokens - 1; 
            // 1. head concat (for the last token only)
            for (int h = 0; h < nof_heads; h++) {
                // More efficient to copy the whole head's output for the token at once
                memcpy(&final_attention_output[i][h * head_dim], &context_heads[h][i][0], head_dim * sizeof(act_t));
            }

            // 2. Attention projection (on the last token only)
            dot_2d(&final_attention_output[i][0],1,d_model,d_model,tbp->attn_proj_weight,d_model,d_model,d_model,&context[i][0],1,d_model,d_model,1,!APPLY_ATTENTION_SCALING);
            
            // Attn projection bias
            add_bias_2d(&context[i][0],1,d_model,tbp->attn_proj_bias,NULL);
            
            // 3. Residual connection
            add_2d(input + (i * d_model),1,d_model,&context[i][0],&residual_out[i][0]);            

            // 4. Layer Norm 2 (on the last token only)
            layernorm_2d(&residual_out[i][0],1,d_model,tbp->ln2_gamma,tbp->ln2_beta, &X_norm2[i][0],eps);
            
            // 5. MLP (on the last token only)
            dot_2d(&X_norm2[i][0],1,d_model,d_model,tbp->W1,d_ff,d_model,d_model,&X1_out[i][0],1,d_ff,d_ff,1,!APPLY_ATTENTION_SCALING);
            // W1 bias
            add_bias_2d(&X1_out[i][0],1,d_ff,tbp->b1,NULL);
            // GELU activation
            gelu_2d(&X1_out[i][0],d_ff,1,NULL);
            // W2 
            dot_2d(&X1_out[i][0],1,d_ff,d_ff,tbp->W2,d_model,d_ff,d_ff,&X2_out[i][0],1,d_model,d_model,1,!APPLY_ATTENTION_SCALING);
            // W2 bias
            add_bias_2d(&X2_out[i][0],1,d_model,tbp->b2,NULL);
            
            // 6. Final Residual Connection (for the last token only)
            // First, preserve the state of previous tokens by copying them over
            if (i > 0) {
                memcpy(&residual2_out[0][0], input, i * d_model * sizeof(act_t));
            }
            // Then, calculate the new residual for the last token
            add_2d(&residual_out[i][0], 1, d_model, &X2_out[i][0], &residual2_out[i][0]);

        } else {
            // prefill phase or non cached path
            for (int i = 0; i < n_tokens; i++) {
                for (int h = 0; h < nof_heads; h++) {
                     memcpy(&final_attention_output[i][h * head_dim], &context_heads[h][i][0], head_dim * sizeof(act_t));
                }            
            }
            // Attention projection 
        dot_2d(&final_attention_output[0][0],n_tokens,d_model,d_model,tbp->attn_proj_weight,d_model,d_model,d_model,&context[0][0],ctx_len,d_model,d_model,1,!APPLY_ATTENTION_SCALING);
        print_2d_tensor("C context[1][:10](before bias):",&context[1][0],ctx_len,d_model,1,10,0);

        // Attn projection bias
        add_bias_2d(&context[0][0],n_tokens,d_model,tbp->attn_proj_bias,NULL);
        print_2d_tensor("C context[1][:10]:",&context[1][0],ctx_len,d_model,1,10,0);

        // Residual connection
        add_2d(input,n_tokens,d_model,&context[0][0],&residual_out[0][0]);
        print_2d_tensor("C context[1][:10]:",&residual_out[1][0],ctx_len,d_model,1,10,0);


        // Layer Norm 2
        layernorm_2d(&residual_out[0][0],n_tokens,d_model,tbp->ln2_gamma,tbp->ln2_beta, &X_norm2[0][0],eps);
        print_2d_tensor("C X_norm2[1][:10]:",&X_norm2[1][0],ctx_len,d_model,1,10,0);


        // MLP layer, W1
        dot_2d(&X_norm2[0][0],n_tokens,d_model,d_model,tbp->W1,d_ff,d_model,d_model,&X1_out[0][0],n_tokens,d_ff,d_ff,1,!APPLY_ATTENTION_SCALING);
        print_2d_tensor("C X1_out[1][:10] before bias:",&X1_out[1][0],ctx_len,d_ff,1,10,0);

        // W1 bias
        add_bias_2d(&X1_out[0][0],n_tokens,d_ff,tbp->b1,NULL);
        print_2d_tensor("C X1_out[1][:10] before bias:",&X1_out[1][0],ctx_len,d_ff,1,10,0);

        // GELU activation
        gelu_2d(&X1_out[0][0],n_tokens,d_ff,NULL);
        print_2d_tensor("C X1_out after GELU[1][:10]",&X1_out[1][0],ctx_len,d_ff,1,10,0);
            
        // W2 
        dot_2d(&X1_out[0][0],n_tokens,d_ff,d_ff,tbp->W2,d_model,d_ff,d_ff,&X2_out[0][0],n_tokens,d_model,d_model,1,!APPLY_ATTENTION_SCALING);
        print_2d_tensor("C X2_out[1][:10] before bias:",&X2_out[1][0],ctx_len,d_model,1,10,0);

        // W2 bias
        add_bias_2d(&X2_out[0][0],n_tokens,d_model,tbp->b2,NULL);
        print_2d_tensor("C X2_out[1][:10]:",&X2_out[1][0],ctx_len,d_model,1,10,0);

        // Residual connection
        add_2d(&X2_out[0][0],n_tokens,d_model,&residual_out[0][0],&residual2_out[0][0]);

    }                    
    json_object_set_new(json_root, layer_key, layer_obj);
}


static int parse_tokens(const char *json, int *tokens, int max_tokens) {
    const char *start = strchr(json, '[');
    const char *end = strchr(json, ']');
    if (!start || !end || start > end) return -1; // Invalid format

    start++; // skip '['
    int count = 0;
    char number[16]; // temporary buffer

    while (start < end && count < max_tokens) {
        // Skip spaces
        while (*start == ' ' || *start == ',') start++;

        int len = 0;
        while (start < end && *start >= '0' && *start <= '9' && len < 15) {
            number[len++] = *start;
            start++;
        }
        number[len] = '\0';

        if (len > 0) {
            
            tokens[count++] = atoi(number);
            //printf("The token inside buffer = %d\n",tokens[count-1]);
            
        }
    }

    return count; // number of tokens parsed
}

#if defined(USE_INT8)
// On-disk alignment of every block written by the offline quant tool.
// Must match tools/offline_quant/src/writer.py ALIGNMENT.
#define QUANT8_BIN_ALIGNMENT 16

static void quant8_skip_pad(size_t bytes_read, FILE *fp) {
    size_t pad = (QUANT8_BIN_ALIGNMENT - (bytes_read % QUANT8_BIN_ALIGNMENT)) % QUANT8_BIN_ALIGNMENT;
    if (pad && fseek(fp, (long)pad, SEEK_CUR) != 0) {
        fprintf(stderr, "Error: fseek over %zu bytes of alignment padding failed.\n", pad);
        exit(1);
    }
}

// Read one INT8 weight block immediately followed by its FP32 per-output-channel
// scale vector. Each block is padded to QUANT8_BIN_ALIGNMENT bytes on disk.
static void fread_int8_with_scale_or_exit(
    qweight_t *dest_int8, size_t int8_count,
    qscale_t  *dest_scale, size_t scale_count,
    FILE *fp)
{
    if (fread(dest_int8, sizeof(qweight_t), int8_count, fp) != int8_count) {
        fprintf(stderr, "Error: fread of %zu int8 weights failed or EOF.\n", int8_count);
        exit(1);
    }
    quant8_skip_pad(int8_count * sizeof(qweight_t), fp);

    if (fread(dest_scale, sizeof(qscale_t), scale_count, fp) != scale_count) {
        fprintf(stderr, "Error: fread of %zu fp32 scales failed or EOF.\n", scale_count);
        exit(1);
    }
    quant8_skip_pad(scale_count * sizeof(qscale_t), fp);
}
#endif

// Read `count` FP32 elements from disk into a weight_t buffer, converting
// element-by-element when the build dtype is narrower than FP32.
// On-disk format is always FP32 — only the in-memory representation changes.
static void fread_weights_or_exit(weight_t *dest, size_t count, FILE *fp) {
#if defined(USE_BF16) || defined(USE_FP16)
    static float chunk[4096];
    const size_t N = sizeof(chunk) / sizeof(chunk[0]);
    while (count > 0) {
        size_t k = (count < N) ? count : N;
        if (fread(chunk, sizeof(float), k, fp) != k) {
            fprintf(stderr, "Error: fread failed or unexpected EOF.\n");
            exit(1);
        }
        for (size_t i = 0; i < k; i++) {
            dest[i] = (weight_t)chunk[i];
        }
        dest += k;
        count -= k;
    }
#else
    if (fread(dest, sizeof(weight_t), count, fp) != count) {
        fprintf(stderr, "Error: fread failed or unexpected EOF.\n");
        exit(1);
    }
#endif
}


static void copy_weights_to_gpu(void){
#ifdef USE_CUDA

    CUDA_CHECK(cudaMemcpy(wte_d,  wte, vocab_size * sizeof(*wte_d),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(wpe_d,  wpe, ctx_len * sizeof(*wpe_d),cudaMemcpyHostToDevice));
    //CUDA_CHECK(cudaMemcpy(wte_T_d,wte_T, sizeof (*wte_T),cudaMemcpyHostToDevice));

#if !defined(USE_INT8)
    /* BF16 GEMM weights — replaced by *_int8_d + *_scale_d under USE_INT8.
     * The host source buffers are not allocated in that build, so skip the copy. */
    CUDA_CHECK(cudaMemcpy(W_q_d,  W_q, num_layers * sizeof (*W_q_d),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(W_k_d,  W_k, num_layers * sizeof (*W_k_d),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(W_v_d,  W_v, num_layers * sizeof (*W_v_d),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(attn_proj_weight_d,  attn_proj_weight, num_layers * sizeof (*attn_proj_weight_d),cudaMemcpyHostToDevice));
#endif
    CUDA_CHECK(cudaMemcpy(attn_proj_bias_d,  attn_proj_bias, num_layers * sizeof (*attn_proj_bias_d),cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(b_q_d,  b_q, num_layers * sizeof (*b_q_d),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_k_d,  b_k, num_layers * sizeof (*b_k_d),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_v_d,  b_v, num_layers * sizeof (*b_v_d),cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(layer_norm1_gamma_d,  layer_norm1_gamma, num_layers * sizeof (*layer_norm1_gamma_d),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(layer_norm1_beta_d,  layer_norm1_beta, num_layers * sizeof (*layer_norm1_beta_d),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(layer_norm2_gamma_d,  layer_norm2_gamma, num_layers * sizeof (*layer_norm2_gamma_d),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(layer_norm2_beta_d,  layer_norm2_beta, num_layers * sizeof (*layer_norm2_beta_d),cudaMemcpyHostToDevice));


#if !defined(USE_INT8)
    /* BF16 W1/W2 — replaced by *_int8_d + *_scale_d under USE_INT8. */
    CUDA_CHECK(cudaMemcpy(W1_d,  W1, num_layers * sizeof (*W1_d),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(W2_d,  W2, num_layers * sizeof (*W2_d),cudaMemcpyHostToDevice));
#endif
    CUDA_CHECK(cudaMemcpy(b1_d,  b1, num_layers * sizeof (*b1_d),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b2_d,  b2, num_layers * sizeof (*b2_d),cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(layer_normf_gamma_d, layer_normf_gamma, d_model * sizeof(weight_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(layer_normf_beta_d, layer_normf_beta, d_model * sizeof(weight_t), cudaMemcpyHostToDevice));

#if defined(USE_INT8)
    /* Copy INT8 weight buffers + their per-output-channel FP32 scales.
     * The BF16 weight cudaMemcpys above remain — they target the BF16
     * device buffers which carry uninitialized data under USE_INT8 and
     * will be skipped once the GEMM call sites are switched to the INT8
     * dispatch in phase 4. */
    CUDA_CHECK(cudaMemcpy(W_q_int8_d,        W_q_int8,        num_layers * sizeof *W_q_int8_d,        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(W_k_int8_d,        W_k_int8,        num_layers * sizeof *W_k_int8_d,        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(W_v_int8_d,        W_v_int8,        num_layers * sizeof *W_v_int8_d,        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(attn_proj_int8_d,  attn_proj_int8,  num_layers * sizeof *attn_proj_int8_d,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(W1_int8_d,         W1_int8,         num_layers * sizeof *W1_int8_d,         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(W2_int8_d,         W2_int8,         num_layers * sizeof *W2_int8_d,         cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(W_q_scale_d,        W_q_scale,        num_layers * sizeof *W_q_scale_d,        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(W_k_scale_d,        W_k_scale,        num_layers * sizeof *W_k_scale_d,        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(W_v_scale_d,        W_v_scale,        num_layers * sizeof *W_v_scale_d,        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(attn_proj_scale_d,  attn_proj_scale,  num_layers * sizeof *attn_proj_scale_d,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(W1_scale_d,         W1_scale,         num_layers * sizeof *W1_scale_d,         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(W2_scale_d,         W2_scale,         num_layers * sizeof *W2_scale_d,         cudaMemcpyHostToDevice));
#endif

    CUDA_CHECK(cudaDeviceSynchronize());

#endif

}

static void load_all_weights(FILE* fp){
    // token + position embeddings — preserved as FP32 on disk under both builds.
    fread_weights_or_exit(&wte[0][0], (size_t)vocab_size*d_model, fp);
    fread_weights_or_exit(&wpe[0][0], (size_t)ctx_len*d_model, fp);

#if defined(USE_INT8)
    /* Per-layer INT8 staging buffer for the packed W_qkv block before
     * splitting into per-row Q/K/V. Each block on disk is INT8 [3*d_model,
     * d_model] followed by FP32 scale[3*d_model], 16-byte aligned. */
    static qweight_t temp_W_qkv_int8[3*d_model][d_model];
    static qscale_t  temp_W_qkv_scale[3*d_model];
#endif

    for (int l = 0; l < num_layers; l++){

        fread_weights_or_exit(layer[l].ln1_gamma, d_model, fp);//ln_1.weight
        fread_weights_or_exit(layer[l].ln1_beta,  d_model, fp);//ln_1.bias

#if defined(USE_INT8)
        /* INT8 W_qkv + scale, then split rows 0..d_model-1 → Q,
         * d_model..2*d_model-1 → K, 2*d_model..3*d_model-1 → V.
         * Same row-split as the FP32 path; scale vector splits the same way. */
        fread_int8_with_scale_or_exit(
            &temp_W_qkv_int8[0][0], (size_t)3*d_model*d_model,
            &temp_W_qkv_scale[0],   (size_t)3*d_model,
            fp);
        for (int i = 0; i < d_model; i++) {
            for (int j = 0; j < d_model; j++) {
                W_q_int8[l][i][j] = temp_W_qkv_int8[i][j];
                W_k_int8[l][i][j] = temp_W_qkv_int8[i + d_model][j];
                W_v_int8[l][i][j] = temp_W_qkv_int8[i + 2*d_model][j];
            }
            W_q_scale[l][i] = temp_W_qkv_scale[i];
            W_k_scale[l][i] = temp_W_qkv_scale[i + d_model];
            W_v_scale[l][i] = temp_W_qkv_scale[i + 2*d_model];
        }
#else
        fread_weights_or_exit(&temp_attn_weight[0][0], (size_t)d_model * 3 * d_model, fp);
        for (int i = 0; i < d_model; i++) {
            for (int j = 0; j < d_model; j++) {
                layer[l].W_q[i * d_model + j] = temp_attn_weight[i][j];               // rows 0–767
                layer[l].W_k[i * d_model + j] = temp_attn_weight[i + d_model][j];     // rows 768–1535
                layer[l].W_v[i * d_model + j] = temp_attn_weight[i + 2 * d_model][j]; // rows 1536–2303
            }
        }
#endif

        /* QKV bias is preserved (FP32 in the .bin under both builds);
         * existing FP32→BF16 path handles the cast. */
        fread_weights_or_exit(temp_attn_bias, 3*d_model, fp);
        for (int i = 0; i < d_model; i++) {
            layer[l].b_q[i] = temp_attn_bias[i];
            layer[l].b_k[i] = temp_attn_bias[d_model + i];
            layer[l].b_v[i] = temp_attn_bias[2*d_model + i];
        }

#if defined(USE_INT8)
        fread_int8_with_scale_or_exit(
            &attn_proj_int8[l][0][0], (size_t)d_model*d_model,
            &attn_proj_scale[l][0],   (size_t)d_model,
            fp);
#else
        fread_weights_or_exit(layer[l].attn_proj_weight, (size_t)d_model*d_model, fp);//attn.c_proj.weight
#endif
        fread_weights_or_exit(layer[l].attn_proj_bias,   d_model, fp);//attn.c_proj.bias

        fread_weights_or_exit(layer[l].ln2_gamma, d_model, fp);//ln_2.weight
        fread_weights_or_exit(layer[l].ln2_beta,  d_model, fp);//ln_2.bias

#if defined(USE_INT8)
        fread_int8_with_scale_or_exit(
            &W1_int8[l][0][0], (size_t)d_ff*d_model,
            &W1_scale[l][0],   (size_t)d_ff,
            fp);
#else
        fread_weights_or_exit(layer[l].W1, (size_t)d_model*d_ff, fp);//mlp.c_fc.weight
#endif
        fread_weights_or_exit(layer[l].b1, d_ff, fp);//mlp.c_fc.bias

#if defined(USE_INT8)
        fread_int8_with_scale_or_exit(
            &W2_int8[l][0][0], (size_t)d_model*d_ff,
            &W2_scale[l][0],   (size_t)d_model,
            fp);
#else
        fread_weights_or_exit(layer[l].W2, (size_t)d_ff*d_model, fp);//mlp.c_proj.weight
#endif
        fread_weights_or_exit(layer[l].b2, d_model, fp);//mlp.c_proj.bias
    }
    // final layer norm
    fread_weights_or_exit(layer_normf_gamma, d_model, fp);
    fread_weights_or_exit(layer_normf_beta,  d_model, fp);

    //printf("before copying weights to the gpu\n");
    // no op if no GPU
    copy_weights_to_gpu();

}

#if 0
static void load_layers_weights(TransformerBlockParams * p_tfb, int layer_id,FILE * fp){
    
    fread_or_exit(p_tfb->ln1_gamma, sizeof(float), d_model, fp);//ln_1.weight (768)
    fread_or_exit(p_tfb->ln1_beta,  sizeof(float), d_model, fp);//ln_1.bias (768)
    fread_or_exit(temp_attn_weight, sizeof(float), d_model * 3 * d_model, fp);

    // Split temp_attn_weight into W_q, W_k, W_v
    for (int i = 0; i < d_model; i++) {
        for (int j = 0; j < d_model; j++) {
            p_tfb->W_q[i * d_model + j] = temp_attn_weight[i][j];               // rows 0–767
            p_tfb->W_k[i * d_model + j] = temp_attn_weight[i + d_model][j];     // rows 768–1535
            p_tfb->W_v[i * d_model + j] = temp_attn_weight[i + 2 * d_model][j]; // rows 1536–2303
        }
    }
 
    fread_or_exit(temp_attn_bias, sizeof(float), 3*d_model, fp);
    for (int i = 0; i < d_model; i++) {
        p_tfb->b_q[i] = temp_attn_bias[i];
        p_tfb->b_k[i] = temp_attn_bias[d_model + i];
        p_tfb->b_v[i] = temp_attn_bias[2*d_model + i];
    }

    fread_or_exit(p_tfb->attn_proj_weight,  sizeof(float), d_model*d_model, fp);//attn.c_proj.weight
    fread_or_exit(p_tfb->attn_proj_bias,  sizeof(float), d_model, fp);//attn.c_proj.bias
    
    fread_or_exit(p_tfb->ln2_gamma, sizeof(float), d_model, fp);//ln_2.weight
    fread_or_exit(p_tfb->ln2_beta,  sizeof(float), d_model, fp);//ln_2.bias

    fread_or_exit(p_tfb->W1, sizeof(float), d_model*d_ff, fp);//mlp.c_fc.weight
    fread_or_exit(p_tfb->b1, sizeof(float), d_ff, fp);//mlp.c_fc.bias
    fread_or_exit(p_tfb->W2, sizeof(float), d_ff*d_model, fp);//mlp.c_proj.weight
    fread_or_exit(p_tfb->b2, sizeof(float), d_model, fp);//mlp.c_proj.bias

}
#endif

#define MAX_OUTPUT_TOKENS 1024
#define CHARS_PER_TOKEN 7
#define MAX_TOKEN_LIST_CHARS (MAX_OUTPUT_TOKENS * CHARS_PER_TOKEN + 10) // 1024 tokens, up to 6 digits + comma/space, plus buffer
#define MAX_PROMPT_BYTES 8192 // input_buffer size — must match the declaration in main()
// Sized to fit either the decode request (tokens->text) or the encode request (text->tokens)
#define MAX_JSON_REQUEST_CHARS ((MAX_TOKEN_LIST_CHARS > MAX_PROMPT_BYTES ? MAX_TOKEN_LIST_CHARS : MAX_PROMPT_BYTES) + 64)

int main(int argc, char *argv[])
{
    const int n_tokens = 1024; // TODO:replace with ctx_len
    char input_buffer[MAX_PROMPT_BYTES];
    char encode_request[MAX_JSON_REQUEST_CHARS];
    char encode_response[MAX_JSON_REQUEST_CHARS];
    char decode_request[MAX_JSON_REQUEST_CHARS];
    char decode_response[MAX_JSON_REQUEST_CHARS];
    int tokens[n_tokens] = {};
    char token_list[MAX_TOKEN_LIST_CHARS] = {0};
    

    //long offset = (vocab_size * d_model + ctx_len * d_model) * sizeof(float);
    char temp_token_str[32];
    
    // Load GTP2 weights
    //printf("Loading GPT2 weights...\n");
    allocate_weights();
    init_layer_table();    

    // 
    // Open the file containing the weights, load all the weights and close it
#if defined(USE_INT8)
    const char *weights_filename = MODEL_QUANT8_WEIGHTS_FILENAME;
#else
    const char *weights_filename = MODEL_WEIGHTS_FILENAME;
#endif
    FILE * fp = fopen(weights_filename,"rb");
    if (!fp) {
        fprintf(stderr, "FATAL: cannot open weights file '%s' (errno %d)\n",
                weights_filename, errno);
        exit(1);
    }
    //printf("filed is opened\n");
    load_all_weights(fp);
    fclose(fp);

    update_layer_table();
    //printf("load weights...");
#ifdef USE_CUDA
    //print_gpu_memory_usage();

#endif

    json_t *json_root = json_object();
    json_t *perf_root   = json_object();  // top level
    json_t *chunk_array = json_array();   // per chunk entries
    json_t *tpot_array  = json_array();   // per-token latencies
    int current_seq_len_initial = 0;
    float total_inference_elapsed = 0;
    struct timespec token_start, token_end;
    double ttft = 0.0;
    json_object_set_new(perf_root, "model_variant",  json_string(MODEL));
    json_object_set_new(perf_root, "kv_cache_enabled", json_integer(kv_cache_enabled)); 

    // Add run timestamp    
    time_t now = time(NULL);
    char time_str[32];
    strftime(time_str, sizeof(time_str), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
    json_object_set_new(perf_root, "run_utc", json_string(time_str));
    // Store array for per-chunk timings
    json_object_set_new(perf_root, "chunks", chunk_array);

    transpose_2d(&wte[0][0], vocab_size,d_model , &wte_T[0][0]);
#ifdef USE_CUDA
    CUDA_CHECK(cudaMemcpy(wte_T_d, wte_T, d_model * sizeof(*wte_T_d), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    //print_gpu_memory_usage();
    //return 1;
#endif

    float temperature = 1.0;
    int requested_out_tokens = 768; // 16, 32, 64, 128, 256, 512, 1024 (measure performance I used 768)
    int token_chunk_size = 32; // set to 1 for latency measurement
    struct timespec loop_start, loop_end; // For per-chunk/per-token timing

    char *cli_input = NULL;
    char *json_out_file = NULL;
    int stream = 1;     // streaming on by default
    int verbose = 0;    // chunk stats off by default

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            cli_input = argv[i + 1];
            i++;
        }else if (strcmp(argv[i], "--req_out_tokens")==0 && i + 1 < argc){
            requested_out_tokens = atoi(argv[i+1]); // fine to use atoi for now.
            i++;
        }else if(strcmp(argv[i], "--token_chunk_size")==0 && i + 1 < argc){
            token_chunk_size = atoi(argv[i+1]); // fine to use atoi for now
            i++;
        }else if(strcmp(argv[i], "--json_out_file")==0 && i + 1 < argc){
            json_out_file = argv[i+1];
            i++;
        }else if(strcmp(argv[i], "--no-stream")==0){
            stream = 0;
        }else if(strcmp(argv[i], "--verbose")==0){
            verbose = 1;
        }
    }
    if (verbose) {
        printf("req_out_tokens = %d\n",requested_out_tokens);
        printf("token_chunk_size = %d\n",token_chunk_size);
        printf("json_out_file = %s\n",json_out_file);
    }

    while(1){ 
        
        if (cli_input){
            strncpy(input_buffer, cli_input, sizeof(input_buffer));
            input_buffer[sizeof(input_buffer) - 1] = '\0';            
        } else {
            printf("Enter Input:");
            if (fgets(input_buffer, sizeof(input_buffer), stdin) == NULL){
                printf("fgets failed, quit\n");
                break;
            };
            input_buffer[strcspn(input_buffer, "\n")] = 0;
            if (strcmp(input_buffer, "q") == 0) {
                printf("QUIT\n");
                break;
            }

            // Prompt for generation parameters
            char param_buf[64];
            printf("max_length [%d]: ", requested_out_tokens);
            if (fgets(param_buf, sizeof(param_buf), stdin) != NULL) {
                param_buf[strcspn(param_buf, "\n")] = 0;
                if (strlen(param_buf) > 0) {
                    requested_out_tokens = atoi(param_buf);
                }
            }
            printf("temperature [%.1f]: ", temperature);
            if (fgets(param_buf, sizeof(param_buf), stdin) != NULL) {
                param_buf[strcspn(param_buf, "\n")] = 0;
                if (strlen(param_buf) > 0) {
                    temperature = atof(param_buf);
                }
            }
        }
        // store prompt in json
        json_object_set_new(perf_root, "prompt", json_string(input_buffer));

        clock_gettime(CLOCK_MONOTONIC, &start); // start the overall inference
        //printf("\nGPT2 Inference - Start\n");

        // cleanup, move to the end (move to function)
        memset(tokens,0,sizeof(tokens));
        memset(embeddings,0,sizeof(embeddings));            
        memset(encode_request, 0, sizeof(encode_request));
        memset(encode_response, 0, sizeof(encode_response));
        memset(decode_request, 0, sizeof(decode_request));
        memset(decode_response, 0, sizeof(decode_response));
        memset(token_list,0,sizeof(token_list));
        // Reset KV caches for each new prompt
#ifdef USE_CUDA
        CUDA_CHECK(cudaMemset(K_cache_d, 0, num_layers * sizeof *K_cache_d));
        CUDA_CHECK(cudaMemset(V_cache_d, 0, num_layers * sizeof *V_cache_d));
#else
        memset(K_cache, 0, sizeof(K_cache));
        memset(V_cache, 0, sizeof(V_cache));
#endif
        // after parsing initial tokens
        // should be reset for each new prompt
        last_index = 0;

        // Format
        snprintf(encode_request, sizeof(encode_request),
         "{\"mode\": \"encode\", \"text\": \"%s\"}", input_buffer);
         // Get tokens
        send_json_to_tokenizer(encode_request, encode_response, sizeof(encode_response));
        //printf("Encode Response: %s\n", encode_response);
        // Format
        int current_seq_len = parse_tokens(encode_response, tokens, ctx_len);
        if (current_seq_len < 0) {
            printf("Failed to parse tokens!\n");
            continue;
        }
        current_seq_len_initial = current_seq_len;
        
        // --- Initialize last_token_position after parsing initial prompt ---
        // This is the index of the last token in the *initial prompt*.
        // It will be updated to point to the *newly added* token in each generation loop.
        int last_token_position = current_seq_len - 1; 

        if (verbose) {
            printf("\n--- Token Generation Performance ---\n");
            printf("Initial prompt length: %d\n", current_seq_len);
            printf("--- Total Time for Single Forward Pass (O(N^2) Measurement) ---\n");
            printf("Context Length | Total Pass Time (s)\n");
            printf("------------------------------------\n");
        }

        clock_gettime(CLOCK_MONOTONIC, &loop_start);
         
        int last_index = 0;

        for (int ii = 0; ii < requested_out_tokens; ii++){
            clock_gettime(CLOCK_MONOTONIC, &token_start);

            // calculate n_new_tokens BEFORE updating last_index
            int n_new_tokens = current_seq_len - last_index;

#ifdef USE_CUDA
            // Only copy the new tokens to GPU
            CUDA_CHECK(cudaMemcpy(&tokens_d[last_index],&tokens[last_index],n_new_tokens*sizeof(int),cudaMemcpyHostToDevice));
            // Only compute embeddings for the new tokens
            embeddings_cuda(wte_d,wpe_d,tokens_d,embeddings_d,last_index,n_new_tokens);
#else
            for (int i=last_index; i<current_seq_len;i++){
                for (int j=0; j<d_model;j++){
                    embeddings[i][j] = wte[tokens[i]][j] + wpe[i][j];
                }
            }
#endif

#ifdef USE_CUDA
            act_t *current_hidden_state_d = &embeddings_d[0][0];
#endif

            // reset file pointer position to load weights from the right position
            //fseek(fp, offset, SEEK_SET);
            // Infer (loop through layers)
            for (int i=0 ; i < num_layers; i++){
                //printf("File position before layer %d = %ld\n", i, ftell(fp));
                //load_layers_weights(&layer, i,fp);
                //printf("nof_tokens=%d\n",n_tokens);
                #ifdef USE_CUDA 
                //TODO: use one pointer here (set it before to the right address)
                    transformer_block(current_hidden_state_d,current_seq_len,n_new_tokens, &layer[i],json_root,i,ii);
                    current_hidden_state_d = &residual2_out_d[0][0];
                #else
                    transformer_block(&embeddings[0][0],current_seq_len,n_new_tokens, &layer[i],json_root,i,ii);

                    if (i < num_layers-1){
                        memcpy(&embeddings[0][0], &residual2_out[0][0], sizeof(float) * current_seq_len * d_model);
                    }
                                                            
                    //printf("*****Layer %d completed******\n",i);

                #endif                                                

            }
            //printf("transformer loop done\n");

            if (kv_cache_enabled){
                last_index = current_seq_len;
            }
            // load weights and bias here
            //fread_or_exit(layer_normf_gamma, sizeof(float), d_model, fp);
            //fread_or_exit(layer_normf_beta, sizeof(float), d_model, fp);
            

#ifdef USE_CUDA 
            //print_2d_tensor("C residual2_out[1][:10]:",&residual2_out[1][0],ctx_len,d_model,1,10,0);

            // Layer Norm final (only the last token needed for next-token prediction)
            {
                int ln_start = last_token_position;
                int ln_rows = 1;
                layernorm_cuda((act_t (*)[d_model])&residual2_out_d[ln_start][0],ln_rows,d_model,&layer_normf_gamma_d[0],&layer_normf_beta_d[0],(act_t (*)[d_model])&Xf_out_d[ln_start][0],eps);

                // get logits (only for the last token)
                dot_2d(&Xf_out_d[ln_start][0],ln_rows,d_model,d_model,&wte_T_d[0][0],d_model,vocab_size,vocab_size,&logits_d[ln_start][0],ln_rows,vocab_size,vocab_size,0,!APPLY_ATTENTION_SCALING);
            }
            //print_2d_tensor("C logits[1][i]",&logits[1][0],ctx_len,vocab_size,1,10,0);

#else
            print_2d_tensor("C residual2_out[1][:10]:",&residual2_out[1][0],ctx_len,d_model,1,10,0);

            // Layer Norm final
            layernorm_2d(&residual2_out[0][0],current_seq_len,d_model,&layer_normf_gamma[0],&layer_normf_beta[0],&Xf_out[0][0],eps);
            print_2d_tensor("C Xf_out[1][:10]:",&Xf_out[1][0],ctx_len,d_model,1,10,0);

            // get logits
            dot_2d(&Xf_out[0][0],current_seq_len,d_model,d_model,&wte_T[0][0],d_model,vocab_size,vocab_size,&logits[0][0],current_seq_len,vocab_size,vocab_size,0,!APPLY_ATTENTION_SCALING);
            print_2d_tensor("C logits[1][i]",&logits[1][0],ctx_len,vocab_size,1,10,0);
#endif
            
            // Compute probabilities
            act_t probs[vocab_size];

#ifdef USE_CUDA
            act_t * logit_row_d = &logits_d[last_token_position][0];
            float effective_temp = (temperature <= 0.0f) ? 1.0f : temperature;
            softmax_cuda(logit_row_d, 1, vocab_size, vocab_size, logit_row_d, effective_temp, /*causal_mask=*/0);

            CUDA_CHECK(cudaMemcpy(probs, logit_row_d, vocab_size*sizeof(act_t), cudaMemcpyDeviceToHost));

#else
            //===========================================SOFTMAX START===================================//
            act_t *logit_row = &logits[last_token_position][0];
            float effective_temp = (temperature <= 0.0f) ? 1.0f : temperature;

            // Softmax with temperature and numerical stability
            float max_logit = -INFINITY;
            for (int i = 0; i < vocab_size; i++) {
                float val = logit_row[i] / effective_temp;
                if (val > max_logit) max_logit = val;
            }


            float sum = 0.0;
            for (int i = 0; i < vocab_size; i++) {
                float val = (logit_row[i] / effective_temp) - max_logit;
                probs[i] = expf(val);
                sum += probs[i];
            }

            // Normalize
            for (int i = 0; i < vocab_size; i++) {
                probs[i] /= sum;
            }
    #endif

            int sampled_token = -1;

            if (temperature <= 0.0f) {
                // Greedy decoding: pick the token with highest probability
                float max_prob = -1.0f;
                for (int i = 0; i < vocab_size; i++) {
                    if (probs[i] > max_prob) {
                        max_prob = probs[i];
                        sampled_token = i;
                    }
                }
            } else {
                // --- Apply Top-K Sampling ---
                int top_k_val = 40;
                int selected_indices[vocab_size];
                act_t selected_probs[vocab_size];

                top_k_sample(probs, vocab_size, top_k_val, selected_indices, selected_probs);

                // --- Now, sample from the top_k_val chosen tokens ---
                float r = (float)rand() / RAND_MAX;
                float cumulative_sampled_probs = 0.0;

                for (int i = 0; i < top_k_val; i++) {
                    cumulative_sampled_probs += selected_probs[i];
                    if (r < cumulative_sampled_probs) {
                        sampled_token = selected_indices[i];
                        break;
                    }
                }

                if (sampled_token == -1 && top_k_val > 0) {
                    sampled_token = selected_indices[top_k_val - 1];
                }
            }

            //// Use sampled_token as your predicted next token
            tokens[current_seq_len++] = sampled_token;
            last_token_position = current_seq_len -1;

            // --- Stream token to stdout ---
            if (stream) {
                char stream_req[128];
                char stream_resp[512];
                snprintf(stream_req, sizeof(stream_req),
                         "{\"mode\": \"decode\", \"tokens\": [%d]}", sampled_token);
                send_json_to_tokenizer(stream_req, stream_resp, sizeof(stream_resp));
                // Parse the "text" field from JSON response
                json_error_t jerr;
                json_t *jresp = json_loads(stream_resp, 0, &jerr);
                if (jresp) {
                    const char *txt = json_string_value(json_object_get(jresp, "text"));
                    if (txt) printf("%s", txt);
                    json_decref(jresp);
                }
                fflush(stdout);
            }

            // --- Per-token timing ---
            clock_gettime(CLOCK_MONOTONIC, &token_end);
            double token_elapsed = (token_end.tv_sec - token_start.tv_sec) + (token_end.tv_nsec - token_start.tv_nsec) / 1e9;

            if (ii == 0) {
                // TTFT: time from inference start to first token produced
                ttft = (token_end.tv_sec - start.tv_sec) + (token_end.tv_nsec - start.tv_nsec) / 1e9;
            }

            // Log per-token latency with context length
            {
                json_t *tok_obj = json_object();
                json_object_set_new(tok_obj, "token_index", json_integer(ii));
                json_object_set_new(tok_obj, "context_len", json_integer(current_seq_len));
                json_object_set_new(tok_obj, "latency_s",   json_real(token_elapsed));
                json_array_append_new(tpot_array, tok_obj);
            }

            // --- Time Measurement and Logging per Chunk ---
            if (((ii + 1) % token_chunk_size == 0) || ((ii + 1) == requested_out_tokens)) {
                clock_gettime(CLOCK_MONOTONIC, &loop_end);
                double chunk_elapsed = (loop_end.tv_sec - loop_start.tv_sec) + (loop_end.tv_nsec - loop_start.tv_nsec) / 1e9;

                if (verbose) {
                    printf("  Generated %d tokens. Total context length: %d. Time for last %d tokens: %.4f seconds. Avg/token (this chunk): %.4f s\n",
                           (ii + 1), current_seq_len, token_chunk_size, chunk_elapsed, chunk_elapsed / token_chunk_size);
                    printf("------------------------------------\n");
                }

                clock_gettime(CLOCK_MONOTONIC, &loop_start); // Reset timer for next chunk

                /* -------- JSON logging -------- */
                json_t *chunk_obj = json_object();
                json_object_set_new(chunk_obj, "generated_so_far", json_integer(ii + 1));
                json_object_set_new(chunk_obj, "total_context",   json_integer(current_seq_len));
                json_object_set_new(chunk_obj, "chunk_seconds",   json_real(chunk_elapsed));
                json_object_set_new(chunk_obj, "avg_sec_per_tok", json_real(chunk_elapsed / token_chunk_size));
                json_array_append_new(chunk_array, chunk_obj);
            }

        }
    
        for (int i = 0; i < current_seq_len; i++) {
            //printf("*******TOKEN = %d *****\n",tokens[i]);
            snprintf(temp_token_str, sizeof(temp_token_str), "%d%s", tokens[i], (i < current_seq_len - 1) ? "," : "");
            strcat(token_list, temp_token_str);
        }

        // Now format the decode_request
        snprintf(decode_request, sizeof(decode_request),
        "{\"mode\": \"decode\", \"tokens\": [%s]}", token_list);


        send_json_to_tokenizer(decode_request, decode_response, sizeof(decode_response));

        if (stream) {
            printf("\n");  // newline after streamed tokens
        } else {
            printf("\nDecode Response: %s\n", decode_response);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        total_inference_elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        if (verbose) {
            printf("\nGPT2 Inference - End (Total Generation Time: %.4f seconds)\n", total_inference_elapsed);
            printf("Average Time per Token (overall generation): %.4f seconds\n", total_inference_elapsed / requested_out_tokens);
        } else {
            printf("[%.1f tokens/s]\n", requested_out_tokens / total_inference_elapsed);
        }
    
        if (cli_input){
            // If input was provided via --prompt, only run once
            break;
        }
    }//main loop

    if (!cli_input) {
        // Interactive mode — no performance log needed
        json_decref(perf_root);
        return 0;
    }

    /* initial prompt length (already known) */
    json_object_set_new(perf_root, "initial_prompt_len", json_integer(current_seq_len_initial)); // capture this earlier
    /* Max output length*/
    json_object_set_new(perf_root, "requested_out_tokens", json_integer(requested_out_tokens)); 

    /* Token chunk size */
    json_object_set_new(perf_root, "token_chunk_size", json_integer(token_chunk_size)); 

    /* full decode text */
    json_object_set_new(perf_root, "generated_text",
                    json_string(decode_response));  // decode_response already null-terminated

    
    json_object_set_new(perf_root, "total_seconds", json_real(total_inference_elapsed));

    // --- Standard LLM inference metrics ---
    json_object_set_new(perf_root, "ttft_s", json_real(ttft));
    double decode_time = total_inference_elapsed - ttft;
    double mean_tpot = (requested_out_tokens > 1) ? decode_time / (requested_out_tokens - 1) : 0.0;
    double tps = (total_inference_elapsed > 0) ? requested_out_tokens / total_inference_elapsed : 0.0;
    json_object_set_new(perf_root, "mean_tpot_s", json_real(mean_tpot));
    json_object_set_new(perf_root, "output_tps", json_real(tps));
    json_object_set_new(perf_root, "e2e_latency_s", json_real(total_inference_elapsed));
    json_object_set_new(perf_root, "per_token_latencies", tpot_array);

    
    const char * filename_to_open;

    if (json_out_file) {
        filename_to_open = json_out_file;
    }else{
        filename_to_open = GPT2_PERFORMANCE_JSON_FILE_NAME;        
    }

    FILE *perf_out = fopen(filename_to_open, "w");

    if(perf_out){
        json_dumpf(perf_root, perf_out, JSON_INDENT(4));
        fclose(perf_out);
        printf("» Performance log written to %s\n", filename_to_open);
    } else{
        perror("Error opening output file");
        exit(EXIT_FAILURE);
    }
    
    json_decref(perf_root);

    return 0;
}
