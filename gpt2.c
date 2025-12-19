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
#include <jansson.h>

#define SERVER_PORT 65432
#define SERVER_IP "127.0.0.1"
#define BUF_SIZE 4096


#ifdef USE_CUDA
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
#else 
    #if defined  (USE_ACCELERATE)
        #include <Accelerate/Accelerate.h>
        #define CBLAS_ROW_MAJOR CblasRowMajor
        #define CBLAS_NO_TRANS CblasNoTrans
    #elif defined (USE_ACCELERATE_X86)
        #include <cblas.h>
    #endif
#endif

//#ifdef USE_ACCELERATE
//#include <Accelerate/Accelerate.h>
//#define CBLAS_ROW_MAJOR CblasRowMajor
//#define CBLAS_NO_TRANS CblasNoTrans
//#endif

#define APPLY_ATTENTION_SCALING (1)
#define PI (3.14159265358979323846)

static void print_2d_tensor(char* name, float *a, int a_r_full_dim, int a_c_full_dim, int r_idx_to_print, int c_idx_to_print, int enable);
static float mean_(float *x, int len);
static float variance_(float *x,int len, float mean);
static void softmax_2d(float *a, int a_r, int a_c, int stride, float * c_out);
static void dot_2d(float *a, // Matrix A (input)
            int a_r, // a rows
            int a_c, // a column
            int lda, // leading dim a
            float*b, // Matrix B (input)
            int b_r, // b rows
            int b_c, // b columns
            int ldb, // leading dim b
            float* c_out, // Matrix C (output)
            int c_r, // c rows
            int c_c, // c columns
            int ldc, // stride c
            int transpose_b, // 
            int apply_attention_scaling 
        );
static void transpose_2d(float *a, int a_r, int a_c, float*b);
static void layernorm_2d(float *a, int a_r, int a_c,float * ln_gamma, float * ln_beta,float * out, float epsilon);
static float gelu(float x);
static void gelu_2d(float *a,int a_c, int a_r, float *out);
static void apply_casual_masking(float * a, int a_c,int size);

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
void top_k_sample(float *probs, int vocab_size, int k,
                  int *top_k_indices_out, float *top_k_probs_out) {

    // Ensure k is within a valid range. If k is too large, it defaults to multinomial sampling.
    if (k <= 0 || k > vocab_size) {
        k = vocab_size;
    }

    // Declare a local (stack-allocated) array of TokenProb structs.
    // This is valid in C99 as a Variable Length Array (VLA) if vocab_size
    // is not a compile-time constant, but for large vocab_size (like 50257),
    // this can cause stack overflow. For robustness, global/static or dynamic
    // allocation (malloc) is usually preferred for very large arrays.
    // However, adhering to the "no dynamic allocation" constraint:
    TokenProb token_probs_list[vocab_size];

    // 1. Populate the list of (probability, original_index) pairs
    for (int i = 0; i < vocab_size; i++) {
        token_probs_list[i].prob = probs[i];
        token_probs_list[i].index = i;
    }

    // 2. Sort the list in descending order of probabilities
    // qsort modifies the array in-place.
    qsort(token_probs_list, vocab_size, sizeof(TokenProb), compareTokenProbs);

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

static void add_tensor_to_layer(json_t *layer_obj, const char *tensor_name, float *a, int a_r, int a_c, int r_idx, int c_idx) {
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



static void send_json_to_tokenizer(const char *json_str, char *response_buf) {
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

    int len = recv(sock, response_buf, BUF_SIZE - 1, 0);
    if (len < 0) {
        perror("Receive failed");
        close(sock);
        exit(1);
    }
    response_buf[len] = '\0';

    close(sock);
}


static void softmax_2d(float *a, int a_r, int a_c,int stride, float * c_out){    
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


static void dot_2d_gpu(float *a,int a_r, int a_c, int lda, float*b,int b_r,int b_c,int ldb, float* c_out,int c_r, int c_c, int ldc, int transpose_b,int apply_attention_scaling ){
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

    // Row-Major to Column-Major trick for cuBLAS
    const cublasOperation_t opB = transpose_b ? CUBLAS_OP_N : CUBLAS_OP_T;
    const cublasOperation_t opA = CUBLAS_OP_T;

    cublasStatus_t stat = cublasSgemm(handle, opB, opA, N, M, K, &alpha,
                                      b, ldb, a, lda, &beta, c_out, ldc);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "FATAL: cublasSgemm failed\n");
        exit(1);
    }
}

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
static void dot_2d_cpu(float *a,int a_r, int a_c, int lda, float*b,int b_r,int b_c,int ldb, float* c_out,int c_r, int c_c, int ldc, int transpose_b,int apply_attention_scaling ){

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

static void dot_2d(float *a,int a_r, int a_c, int lda, float*b,int b_r,int b_c,int ldb, float* c_out,int c_r, int c_c, int ldc, int transpose_b,int apply_attention_scaling ){

#ifdef USE_CUDA
    dot_2d_gpu(a,a_r, a_c, lda, b, b_r, b_c, ldb,  c_out, c_r,  c_c,  ldc,  transpose_b, apply_attention_scaling);
#elif defined (USE_ACCELERATE) || defined (USE_ACCELERATE_X86)
    dot_2d_cpu(a,a_r, a_c, lda, b, b_r, b_c, ldb,  c_out, c_r,  c_c,  ldc,  transpose_b, apply_attention_scaling);

#else    
    #error No backend (USE_CUDA or USE_ACCELERATE) is defined for dot_2d!
#endif
}


static void apply_casual_masking(float * a, int a_c, int size){
    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            a[i * a_c + j] = -INFINITY;
        }
    }
}

//prints start from 0 both for row and column
static void print_2d_tensor(char* name, float *a, int a_r_full_dim, int a_c_full_dim, int r_idx_to_print, int c_idx_to_print, int enable) {
    if (!enable) {
        return;
    }

    printf("--- Tensor: %s ---\n", name);
    printf("[");
    for (int i = 0; i < r_idx_to_print; i++) { // Loop for rows to print
        printf("  [");
        for (int j = 0; j < c_idx_to_print; j++) { // Loop for columns to print
            // Access element using the *full physical column dimension* (a_c_full_dim) as the stride
            printf(" %6.4f", *(a + i * a_c_full_dim + j));
            if (j < c_idx_to_print - 1) printf(", ");
        }
        printf("]");
        if (i < r_idx_to_print - 1) printf(",\n");
        else printf("\n");
    }
    printf("]\n");
}


static void transpose_2d(float *a, int a_r, int a_c, float*b){
// TODO input check
    for (int i=0; i<a_r; i++){
        for (int j=0; j<a_c; j++){
            *(b + j*a_r +i) = *(a + i*a_c + j);
        }
    }
}

static void layernorm_2d(float *a, int a_r, int a_c, 
                    float * ln_gamma, float * ln_beta,
                    float * out, float epsilon){
    
    for (int i=0; i < a_r; i++){
        float *row = a + i * a_c;
        float mean = mean_(row,a_c);
        float var = variance_(row,a_c,mean);
        for (int j=0; j<a_c; j++){
            *(out + i*a_c + j) = *(ln_gamma+j) * ((*(a + i*a_c + j) - mean)/sqrt(var+epsilon)) + *(ln_beta+j);
        }
    }
}

static void add_2d(float *a, int a_r, int a_c, float *b, float *out){
    for (int i=0; i<a_r; i++){
        for (int j=0; j<a_c; j++){
            *(out +i*a_c + j) = *(a +i*a_c + j) + *(b +i*a_c + j);
        }
    }
}

// if out is null addition is inplace 
static void add_bias_2d(float *a, int a_r, int a_c, float *b, float *out){
    float * tmp = out;
    if (out == NULL){ //inplace
        tmp = a;
    } 
    for (int i=0; i<a_r; i++){
        for (int j=0; j<a_c; j++){
            *(tmp +i*a_c + j) = *(a +i*a_c + j) + *(b + j);
        }
    }
}

static float mean_(float *x, int len){
    float sum = 0.0;
    for (int i=0; i<len; i++){
        sum += *(x+i);
    }
    return (sum/(float)len);
}
static float variance_(float *x,int len, float mean){
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

static void gelu_2d(float *a,int a_c, int a_r, float *out){
    float * tmp = out;
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

#define vocab_size (50257)
#define ctx_len (1024) 

#if !defined(GPT2_SMALL_MODEL) && !defined(GPT2_MEDIUM_MODEL) && !defined(GPT2_LARGE_MODEL)
    #define GPT2_MEDIUM_MODEL
#endif

#ifdef GPT2_SMALL_MODEL
    #define  d_model (768)
    #define num_layers (12)
    #define nof_heads  (12)
    #define MODEL_WEIGHTS_FILENAME "weights/gpt2_c_weights.bin"
    #define GPT2_PERFORMANCE_JSON_FILE_NAME "./logs/gpt2_small_performance.json"
    #define MODEL "gpt2_small"
#elif defined(GPT2_MEDIUM_MODEL)
    #define d_model (1024) // GPT-2 Medium
    #define num_layers (24) // GPT-2 Medium
    #define nof_heads (16) // GPT-2 Medium
    /// TODO - to update the readme file or to remove the weights dir since it was missing
    #define MODEL_WEIGHTS_FILENAME "weights/gpt2_medium_c_weights.bin"
    #define GPT2_PERFORMANCE_JSON_FILE_NAME "./logs/gpt2_medium_performance.json"
    #define MODEL "gpt2_medium"
#elif defined(GPT2_LARGE_MODEL)
    #define d_model (1280) // GPT-2 Large
    #define num_layers (36) // GPT-2 Large
    #define nof_heads (20) // GPT-2 Large
    #define MODEL_WEIGHTS_FILENAME "weights/gpt2_large_c_weights.bin"
    #define GPT2_PERFORMANCE_JSON_FILE_NAME "./logs/gpt2_large_performance.json"
    #define MODEL "gpt2_large"

#else
    #error "No GPT-2 model size defined!"
#endif


#ifdef ENABLE_KV_CACHE
int kv_cache_enabled = 1;
#else
int kv_cache_enabled = 0;
#endif

const float eps = 0.00001;
#define head_dim  (d_model/nof_heads)
#define d_ff  (d_model * 4)

float wte[vocab_size][d_model] = {};
float wte_T[d_model][vocab_size] = {};

float wpe[ctx_len][d_model] = {};
float embeddings[ctx_len][d_model] = {}; // for now post positional embeddings. This would go into layer norm

float X_norm[ctx_len][d_model] = {};
float X_norm2[ctx_len][d_model] = {};


float X1[ctx_len][d_ff] = {};
float X1_out[ctx_len][d_ff] = {};
float X2[ctx_len][d_model] = {};
float X2_out[ctx_len][d_model] = {};
float Xf_out[ctx_len][d_model] = {};

/***  Attention (per-layer) ***/
float (*W_q)[d_model][d_model];// 
float (*W_k)[d_model][d_model]; 
float (*W_v)[d_model][d_model];
float (*b_q)[d_model];
float (*b_k)[d_model];
float (*b_v)[d_model];

/* Output projections */
float (*attn_proj_weight)[d_model][d_model];
float (*attn_proj_bias)[d_model];

/* Feed Forward */
float (*W1)[d_ff][d_model];
float (*b1)[d_ff];
float (*W2)[d_model][d_ff];
float (*b2)[d_model];

/* Layer Norm */
float (*layer_norm1_gamma)[d_model];
float (*layer_norm1_beta)[d_model];
float (*layer_norm2_gamma)[d_model];
float (*layer_norm2_beta)[d_model];  

#ifdef USE_CUDA
/***  Global weights ***/
float (*wte_d)[d_model];
float (*wte_T_d)[d_model];
float (*wpe_d)[d_model];

/***  Attention (per-layer) ***/
float (*W_q_d)[d_model][d_model];// 
float (*W_k_d)[d_model][d_model]; 
float (*W_v_d)[d_model][d_model];
float (*b_q_d)[d_model];
float (*b_k_d)[d_model];
float (*b_v_d)[d_model];

/* Output projections */
float (*attn_proj_weight_d)[d_model][d_model];
float (*attn_proj_bias_d)[d_model];

/* Feed Forward */
float (*W1_d)[d_ff][d_model];
float (*b1_d)[d_ff];
float (*W2_d)[d_model][d_ff];
float (*b2_d)[d_model];

/* Layer Norm */
float (*layer_norm1_gamma_d)[d_model];
float (*layer_norm1_beta_d)[d_model];
float (*layer_norm2_gamma_d)[d_model];
float (*layer_norm2_beta_d)[d_model];  

float *layer_normf_gamma_d;       // Final Layer Norm Gamma [d_model]
float *layer_normf_beta_d;        // Final Layer Norm Beta [d_model]


#endif

float temp_attn_weight[3*d_model][d_model] = {}; // 2304 = 768 * 3
float temp_attn_bias[3*d_model] = {};

float Q[ctx_len][d_model] = {};
float K[ctx_len][d_model] = {};
//float K_cache[ctx_len][d_model] = {};
float K_cache[num_layers][ctx_len][d_model] = {};

float K_T[d_model][ctx_len] = {};
float V[ctx_len][d_model] = {};
//float V_cache[ctx_len][d_model] = {};
float V_cache[num_layers][ctx_len][d_model] = {};
float attention_scores[ctx_len][ctx_len] = {};
float attention_scores_temp[ctx_len][ctx_len] = {};
float attention_weights[ctx_len][ctx_len] = {};
float context[ctx_len][d_model] = {};


float layer_normf_gamma[d_model] = {}; // default: no scaling
float layer_normf_beta[d_model] = {};  // default: no shifting

float residual_out[ctx_len][d_model] = {};
float residual2_out[ctx_len][d_model] = {};

float logits[ctx_len][vocab_size] = {};

float context_heads[nof_heads][ctx_len][head_dim] = {};
float scores_h[ctx_len][ctx_len] = {};
float weights_h[ctx_len][ctx_len] = {};
float final_attention_output[ctx_len][d_model] = {}; 


typedef struct{
    float * W_q;
    float * W_k;
    float * W_v;
    float * b_q;
    float * b_k;
    float * b_v;
    float *attn_proj_weight;
    float *attn_proj_bias;
    float * W1;
    float * W2;
    float * b1;
    float * b2;
    float *ln1_gamma;
    float *ln1_beta;
    float *ln2_gamma;
    float *ln2_beta;

}TransformerBlockParams;

static TransformerBlockParams layer[num_layers];

#ifdef USE_CUDA
static void allocate_weights_gpu(void){
    cudaError_t err;
    err = cudaMalloc((void**)&wte_d, vocab_size * sizeof *wte_d);
    err = cudaMalloc((void**)&wpe_d, ctx_len * sizeof *wpe_d);
    err = cudaMalloc((void**)&wte_T_d, d_model * sizeof *wte_T_d);

    err = cudaMalloc((void**)&W_q_d, num_layers * sizeof *W_q_d);
    err = cudaMalloc((void**)&W_k_d,num_layers * sizeof *W_k_d);
    err = cudaMalloc((void**)&W_v_d,num_layers * sizeof *W_v_d);
    err = cudaMalloc((void**)&b_q_d,num_layers * sizeof *b_q_d);
    err = cudaMalloc((void**)&b_k_d,num_layers * sizeof *b_k_d);
    err = cudaMalloc((void**)&b_v_d,num_layers * sizeof *b_v_d);
    err = cudaMalloc((void**)&attn_proj_weight_d,num_layers * sizeof *attn_proj_weight_d);
    err = cudaMalloc((void**)&attn_proj_bias_d,num_layers * sizeof *attn_proj_bias_d);
    err = cudaMalloc((void**)&W1_d,num_layers * sizeof *W1_d);
    err = cudaMalloc((void**)&W2_d,num_layers * sizeof *W2_d);
    err = cudaMalloc((void**)&b1_d,num_layers * sizeof *b1_d);
    err = cudaMalloc((void**)&b2_d,num_layers * sizeof *b2_d);
    err = cudaMalloc((void**)&layer_norm1_gamma_d,num_layers * sizeof *layer_norm1_gamma_d);
    err = cudaMalloc((void**)&layer_norm1_beta_d,num_layers * sizeof *layer_norm1_beta_d);
    err = cudaMalloc((void**)&layer_norm2_gamma_d,num_layers * sizeof *layer_norm2_gamma_d);
    err = cudaMalloc((void**)&layer_norm2_beta_d,num_layers * sizeof *layer_norm2_beta_d);

    err = cudaMalloc((void**)&layer_normf_gamma_d, d_model * sizeof(float));
    err = cudaMalloc((void**)&layer_normf_beta_d, d_model * sizeof(float));

}
#endif
static void allocate_weights_cpu(void){
    size_t allocated_size = 0;
    allocated_size += num_layers * sizeof *W_q;
    W_q   =  malloc(num_layers * sizeof *W_q);

    allocated_size += num_layers * sizeof *W_k;
    W_k   =  malloc(num_layers * sizeof *W_k);

    allocated_size += num_layers * sizeof *W_v;
    W_v   =  malloc(num_layers * sizeof *W_v);

    allocated_size += num_layers * sizeof *b_q;
    b_q   =  malloc(num_layers * sizeof *b_q);

    allocated_size += num_layers * sizeof *b_k;
    b_k   =  malloc(num_layers * sizeof *b_k);

    allocated_size += num_layers * sizeof *b_v;
    b_v   =  malloc(num_layers * sizeof *b_v);

    allocated_size += num_layers * sizeof *attn_proj_weight;
    attn_proj_weight = malloc(num_layers * sizeof *attn_proj_weight);

    allocated_size += num_layers * sizeof *attn_proj_bias;
    attn_proj_bias   = malloc(num_layers * sizeof *attn_proj_bias);

    allocated_size += num_layers * sizeof *W1;
    W1  = malloc(num_layers * sizeof *W1);

    allocated_size += num_layers * sizeof *W2;
    W2  = malloc(num_layers * sizeof *W2);

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
    printf("Total allocated memory = %zu\n",allocated_size);
                                
    if (!W_q || !W_k) { perror("malloc"); exit(1); }
    
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
    }
#endif
}
static void allocate_weights(void){
#ifdef USE_CUDA
    allocate_weights_gpu();
    // easiest for now,used to copy weights to GPU memory
    //allocate_weights_cpu();
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
static void transformer_block(float *input,int n_tokens,int n_new_tokens,
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
        float* k_cache_ptr = &K_cache[layer_id][cache_start_index][0];
        dot_2d(&X_norm[cache_start_index][0],n_new_tokens,d_model,d_model,tbp->W_k,d_model,d_model,d_model,k_cache_ptr,n_new_tokens,d_model,d_model,1,!APPLY_ATTENTION_SCALING);
        add_bias_2d(k_cache_ptr,n_new_tokens,d_model,tbp->b_k,NULL);
        //memcpy(&K_cache[layer_id][cache_start_index][0],&K[0][0], n_new_tokens * d_model * sizeof(float));

        float* v_cache_ptr = &V_cache[layer_id][cache_start_index][0];
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
        float *q_h;
        float *k_h;
        float *v_h;
        float *context_h_out = &context_heads[h][0][0];

        if (kv_cache_enabled){
            k_h = &K_cache[layer_id][0][0]+ h * head_dim;
            v_h = &V_cache[layer_id][0][0]+ h * head_dim;
            //context_h_out = &context_heads[h][0][0];

            if(n_new_tokens == 1){
                float* q_last_token_h = &Q[n_tokens - 1][0] + h * head_dim;
                float* scores_last_row = &scores_h[n_tokens - 1][0]; 
                float* weights_last_row = &weights_h[n_tokens - 1][0];
                float* context_last_row = context_h_out + (n_tokens - 1) * head_dim;

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
            memset(scores_h, 0, sizeof(float) * ctx_len * ctx_len);

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
                memcpy(&final_attention_output[i][h * head_dim], &context_heads[h][i][0], head_dim * sizeof(float));
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
                memcpy(&residual2_out[0][0], input, i * d_model * sizeof(float));
            }
            // Then, calculate the new residual for the last token
            add_2d(&residual_out[i][0], 1, d_model, &X2_out[i][0], &residual2_out[i][0]);

        } else {
            // prefill phase or non cached path
            for (int i = 0; i < n_tokens; i++) {
                for (int h = 0; h < nof_heads; h++) {
                     memcpy(&final_attention_output[i][h * head_dim], &context_heads[h][i][0], head_dim * sizeof(float));
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

static void fread_or_exit(void *ptr, size_t size, size_t count, FILE *fp) {
    if (fread(ptr, size, count, fp) != count) {
        fprintf(stderr, "Error: fread failed or unexpected EOF.\n");
        exit(1);
    }
}

static void copy_weights_to_gpu(void){
#ifdef USE_CUDA

    cudaMemcpy(wte_d,  wte, vocab_size * sizeof(*wte_d),cudaMemcpyHostToDevice);
    cudaMemcpy(wpe_d,  wpe, ctx_len * sizeof(*wpe_d),cudaMemcpyHostToDevice);
    //cudaMemcpy(wte_T_d,wte_T, sizeof (*wte_T),cudaMemcpyHostToDevice);

    cudaMemcpy(W_q_d,  W_q, num_layers * sizeof (*W_q_d),cudaMemcpyHostToDevice);
    cudaMemcpy(W_k_d,  W_k, num_layers * sizeof (*W_k_d),cudaMemcpyHostToDevice);
    cudaMemcpy(W_v_d,  W_v, num_layers * sizeof (*W_v_d),cudaMemcpyHostToDevice);

    cudaMemcpy(attn_proj_weight_d,  attn_proj_weight, num_layers * sizeof (*attn_proj_weight_d),cudaMemcpyHostToDevice);
    cudaMemcpy(attn_proj_bias_d,  attn_proj_bias, num_layers * sizeof (*attn_proj_bias_d),cudaMemcpyHostToDevice);

    cudaMemcpy(b_q_d,  b_q, num_layers * sizeof (*b_q_d),cudaMemcpyHostToDevice);
    cudaMemcpy(b_k_d,  b_k, num_layers * sizeof (*b_k_d),cudaMemcpyHostToDevice);
    cudaMemcpy(b_v_d,  b_v, num_layers * sizeof (*b_v_d),cudaMemcpyHostToDevice);

    cudaMemcpy(layer_norm1_gamma_d,  layer_norm1_gamma, num_layers * sizeof (*layer_norm1_gamma_d),cudaMemcpyHostToDevice);
    cudaMemcpy(layer_norm1_beta_d,  layer_norm1_beta, num_layers * sizeof (*layer_norm1_beta_d),cudaMemcpyHostToDevice);
    cudaMemcpy(layer_norm2_gamma_d,  layer_norm2_gamma, num_layers * sizeof (*layer_norm2_gamma_d),cudaMemcpyHostToDevice);
    cudaMemcpy(layer_norm2_beta_d,  layer_norm2_beta, num_layers * sizeof (*layer_norm2_beta_d),cudaMemcpyHostToDevice);


    cudaMemcpy(W1_d,  W1, num_layers * sizeof (*W1_d),cudaMemcpyHostToDevice);
    cudaMemcpy(b1_d,  b1, num_layers * sizeof (*b1_d),cudaMemcpyHostToDevice);
    cudaMemcpy(W2_d,  W2, num_layers * sizeof (*W2_d),cudaMemcpyHostToDevice);
    cudaMemcpy(b2_d,  b2, num_layers * sizeof (*b2_d),cudaMemcpyHostToDevice);

    cudaMemcpy(layer_normf_gamma_d, layer_normf_gamma, d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(layer_normf_beta_d, layer_normf_beta, d_model * sizeof(float), cudaMemcpyHostToDevice);
#endif

}
static void load_all_weights(FILE* fp){
    // token + position embeddings 
    fread_or_exit(wte,sizeof(float),vocab_size*d_model,fp);
    fread_or_exit(wpe,sizeof(float),ctx_len*d_model,fp);

    for (int l = 0; l < num_layers; l++){
        fread_or_exit(layer[l].ln1_gamma, sizeof(float), d_model, fp);//ln_1.weight (768)
        fread_or_exit(layer[l].ln1_beta,  sizeof(float), d_model, fp);//ln_1.bias (768)
        fread_or_exit(temp_attn_weight, sizeof(float), d_model * 3 * d_model, fp);

        for (int i = 0; i < d_model; i++) {
            for (int j = 0; j < d_model; j++) {
                layer[l].W_q[i * d_model + j] = temp_attn_weight[i][j];               // rows 0–767
                layer[l].W_k[i * d_model + j] = temp_attn_weight[i + d_model][j];     // rows 768–1535
                layer[l].W_v[i * d_model + j] = temp_attn_weight[i + 2 * d_model][j]; // rows 1536–2303
            }
        }
        fread_or_exit(temp_attn_bias, sizeof(float), 3*d_model, fp);
        for (int i = 0; i < d_model; i++) {
            layer[l].b_q[i] = temp_attn_bias[i];
            layer[l].b_k[i] = temp_attn_bias[d_model + i];
            layer[l].b_v[i] = temp_attn_bias[2*d_model + i];
        }
        fread_or_exit(layer[l].attn_proj_weight,  sizeof(float), d_model*d_model, fp);//attn.c_proj.weight
        fread_or_exit(layer[l].attn_proj_bias,  sizeof(float), d_model, fp);//attn.c_proj.bias
        
        fread_or_exit(layer[l].ln2_gamma, sizeof(float), d_model, fp);//ln_2.weight
        fread_or_exit(layer[l].ln2_beta,  sizeof(float), d_model, fp);//ln_2.bias

        fread_or_exit(layer[l].W1, sizeof(float), d_model*d_ff, fp);//mlp.c_fc.weight
        fread_or_exit(layer[l].b1, sizeof(float), d_ff, fp);//mlp.c_fc.bias
        fread_or_exit(layer[l].W2, sizeof(float), d_ff*d_model, fp);//mlp.c_proj.weight
        fread_or_exit(layer[l].b2, sizeof(float), d_model, fp);//mlp.c_proj.bias
    }
    // final layer norm
    fread_or_exit(layer_normf_gamma, sizeof(float), d_model, fp);
    fread_or_exit(layer_normf_beta, sizeof(float), d_model, fp);

    // no op if no GPU
    copy_weights_to_gpu();

}

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
#define MAX_OUTPUT_TOKENS 1024 
#define CHARS_PER_TOKEN 7
#define MAX_TOKEN_LIST_CHARS (MAX_OUTPUT_TOKENS * CHARS_PER_TOKEN + 10) // 1024 tokens, up to 6 digits + comma/space, plus buffer
#define MAX_JSON_REQUEST_CHARS (MAX_TOKEN_LIST_CHARS + 64) // For "\{\"mode\": \"decode\", \"tokens\": []}" wrapper

int main(int argc, char *argv[])
{
    const int n_tokens = 1024;
    char input_buffer[2048];    
    char encode_request[MAX_JSON_REQUEST_CHARS];
    char encode_response[MAX_JSON_REQUEST_CHARS];
    char decode_request[MAX_JSON_REQUEST_CHARS];
    char decode_response[MAX_JSON_REQUEST_CHARS];
    int tokens[n_tokens] = {};
    char token_list[MAX_TOKEN_LIST_CHARS] = {0};
    

    //long offset = (vocab_size * d_model + ctx_len * d_model) * sizeof(float);
    char temp_token_str[32];
    
    // Load GTP2 weights
    printf("Loading GPT2 weights...\n");
#ifdef USE_CUDA
    print_gpu_memory_usage();
#endif
    allocate_weights();
    init_layer_table();
    printf("after layer table init\n");
    // no op if CPU
    update_layer_table();
#ifdef USE_CUDA
    print_gpu_memory_usage();
    return 1;
#endif
    // 
    // Open the file containing the weights, load all the weights and close it
    FILE * fp = fopen(MODEL_WEIGHTS_FILENAME,"rb");
    printf("filed is opened\n");
    load_all_weights(fp);
    fclose(fp);

    printf("load weights...");

    json_t *json_root = json_object();
    json_t *perf_root   = json_object();  // top level
    json_t *chunk_array = json_array();   // per chunk entries 
    int current_seq_len_initial = 0;
    float total_inference_elapsed = 0;
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
    // Copy now to GPU
    cudaMemcpy(wte_T_d, wte_T, d_model * sizeof(*wte_T_d), cudaMemcpyHostToDevice);
#endif

    float temperature = 1.0;
    int requested_out_tokens = 768; // 16, 32, 64, 128, 256, 512, 1024 (measure performance I used 768)
    int token_chunk_size = 32; // set to 1 for latency measurement
    struct timespec loop_start, loop_end; // For per-chunk/per-token timing

    char *cli_input = NULL;
    char *json_out_file = NULL;

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
        }
    }
    printf("req_out_tokens = %d\n",requested_out_tokens);
    printf("token_chunk_size = %d\n",token_chunk_size);
    printf("json_out_file = %s\n",json_out_file);

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
            
        }
        // store prompt in json
        json_object_set_new(perf_root, "prompt", json_string(input_buffer));

        clock_gettime(CLOCK_MONOTONIC, &start); // start the overall inference
        printf("\nGPT2 Inference - Start\n");

        // cleanup, move to the end (move to function)
        memset(tokens,0,sizeof(tokens));
        memset(embeddings,0,sizeof(embeddings));            
        memset(encode_request, 0, sizeof(encode_request));
        memset(encode_response, 0, sizeof(encode_response));
        memset(decode_request, 0, sizeof(decode_request));
        memset(decode_response, 0, sizeof(decode_response));
        memset(token_list,0,sizeof(token_list));
        // after parsing initial tokens
        // should be reset for each new prompt
        last_index = 0;

        // Format
        snprintf(encode_request, sizeof(encode_request),
         "{\"mode\": \"encode\", \"text\": \"%s\"}", input_buffer);
         // Get tokens
        send_json_to_tokenizer(encode_request, encode_response);
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

        printf("\n--- Token Generation Performance ---\n");
        printf("Initial prompt length: %d\n", current_seq_len);

        clock_gettime(CLOCK_MONOTONIC, &loop_start);

        printf("--- Total Time for Single Forward Pass (O(N^2) Measurement) ---\n");
        printf("Context Length | Total Pass Time (s)\n");
        printf("------------------------------------\n");

        for (int ii = 0; ii < requested_out_tokens; ii++){  
            
            for (int i=0; i<current_seq_len;i++){
                for (int j=0; j<d_model;j++){
                    // can optimize here
                    embeddings[i][j] = wte[tokens[i]][j] + wpe[i][j];
                }
            }

            // calculate n_new_tokens once before iterating through layers
            int n_new_tokens = current_seq_len - last_index;

            // reset file pointer position to load weights from the right position
            //fseek(fp, offset, SEEK_SET);
            // Infer (loop through layers)
            for (int i=0 ; i < num_layers; i++){
                //printf("File position before layer %d = %ld\n", i, ftell(fp));
                //load_layers_weights(&layer, i,fp);
                //printf("nof_tokens=%d\n",n_tokens);
                                                
                transformer_block(&embeddings[0][0],current_seq_len,n_new_tokens, &layer[i],json_root,i,ii);
                
                // Zero out entire embeddings buffer (safe reset)
                memset(&embeddings[0][0], 0, sizeof(float) * ctx_len * d_model);
                // use the output from this block as the input to the next block
                memcpy(&embeddings[0][0], &residual2_out[0][0], sizeof(float) * current_seq_len * d_model);
                //printf("*****Layer %d completed******\n",i);

            }
            //printf("transformer loop done\n");

            if (kv_cache_enabled){
                last_index = current_seq_len;
            }
            // load weights and bias here
            //fread_or_exit(layer_normf_gamma, sizeof(float), d_model, fp);
            //fread_or_exit(layer_normf_beta, sizeof(float), d_model, fp);
            

            print_2d_tensor("C residual2_out[1][:10]:",&residual2_out[1][0],ctx_len,d_model,1,10,0);

            // Layer Norm final
            layernorm_2d(&residual2_out[0][0],current_seq_len,d_model,&layer_normf_gamma[0],&layer_normf_beta[0],&Xf_out[0][0],eps);
            print_2d_tensor("C Xf_out[1][:10]:",&Xf_out[1][0],ctx_len,d_model,1,10,0);

            // get logits
            dot_2d(&Xf_out[0][0],current_seq_len,d_model,d_model,&wte_T[0][0],d_model,vocab_size,vocab_size,&logits[0][0],current_seq_len,vocab_size,vocab_size,0,!APPLY_ATTENTION_SCALING);
            print_2d_tensor("C logits[1][i]",&logits[1][0],ctx_len,vocab_size,1,10,0);


            //===========================================SOFTMAX START===================================//
            // --- Search the top 5 ---
            float *logit_row = &logits[last_token_position][0];

            // Softmax with temperature and numerical stability
            float max_logit = -INFINITY;
            for (int i = 0; i < vocab_size; i++) {
                float val = logit_row[i] / temperature;
                if (val > max_logit) max_logit = val;
            }

            // Compute probabilities
            float probs[vocab_size];
            float sum = 0.0;
            for (int i = 0; i < vocab_size; i++) {
                float val = (logit_row[i] / temperature) - max_logit;
                probs[i] = expf(val);
                sum += probs[i];
            }

            // Normalize
            for (int i = 0; i < vocab_size; i++) {
                probs[i] /= sum;
            }
            //printf("softmax done");
            //===========================================SOFTMAX END===================================//

            //===========================================MULTINOMIAL Sampling START===================================//
            // Sample from probability distribution
            //float r = (float)rand() / RAND_MAX;
            //float cumulative = 0.0;
            //int sampled_token = -1;
            //for (int i = 0; i < vocab_size; i++) {
            //    cumulative += probs[i];
            //    if (r < cumulative) {
            //        sampled_token = i;
            //        break;
            //    }
            //}
            //===========================================MULTINOMIAL Sampling END===================================//
            
            // --- Apply Top-K Sampling ---
            int top_k_val = 40; // Choose your desired K value (e.g., 40, 50, 100)
            // Ensure these arrays are large enough to hold 'top_k_val' elements
            int selected_indices[vocab_size]; // Use vocab_size as max possible K, or pass actual max_k
            float selected_probs[vocab_size]; // Same here

            //printf("top-k-sample\n");
            top_k_sample(probs, vocab_size, top_k_val, selected_indices, selected_probs);

            // --- Now, sample from the top_k_val chosen tokens ---
            float r = (float)rand() / RAND_MAX;
            float cumulative_sampled_probs = 0.0;
            int sampled_token = -1; // This will store the final token index

            for (int i = 0; i < top_k_val; i++) { // Loop only through the top K tokens
                cumulative_sampled_probs += selected_probs[i];
                if (r < cumulative_sampled_probs) {
                    sampled_token = selected_indices[i]; // Use the index from the top_k_indices array
                    break;
                }
            }

            // If r is very close to 1.0 due to float precision, it might exceed cumulative_sampled_probs in rare cases.
            // Fallback to the last token in the list if no token is explicitly sampled.
            if (sampled_token == -1 && top_k_val > 0) {
                sampled_token = selected_indices[top_k_val - 1];
            }

            //// Use sampled_token as your predicted next token
            tokens[current_seq_len++] = sampled_token;
            last_token_position = current_seq_len -1;
            //printf("%d ",sampled_token);

            // --- Time Measurement and Logging per Chunk ---
            if (((ii + 1) % token_chunk_size == 0) || ((ii + 1) == requested_out_tokens)) {
                clock_gettime(CLOCK_MONOTONIC, &loop_end);
                double chunk_elapsed = (loop_end.tv_sec - loop_start.tv_sec) + (loop_end.tv_nsec - loop_start.tv_nsec) / 1e9;
                printf("  Generated %d tokens. Total context length: %d. Time for last %d tokens: %.4f seconds. Avg/token (this chunk): %.4f s\n", 
                       (ii + 1), current_seq_len, token_chunk_size, chunk_elapsed, chunk_elapsed / token_chunk_size);

                printf("------------------------------------\n"); // Separator for chunks

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


        send_json_to_tokenizer(decode_request, decode_response);
        printf("\nDecode Response: %s\n", decode_response);


        clock_gettime(CLOCK_MONOTONIC, &end);
        total_inference_elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("\nGPT2 Inference - End (Total Generation Time: %.4f seconds)\n", total_inference_elapsed);
        printf("Average Time per Token (overall generation): %.4f seconds\n", total_inference_elapsed / requested_out_tokens);
    
        if (cli_input){
            // If input was provided via --prompt, only run once
            break;
        }
    }//main loop

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

    
    const char * filename_to_open;

    if (json_out_file) {
        filename_to_open = json_out_file;
    }else{
        filename_to_open = GPT2_PERFORMANCE_JSON_FILE_NAME;        
    }

    FILE *perf_out = fopen(filename_to_open, "w");

    if(perf_out){
        json_dumpf(perf_root, perf_out, JSON_INDENT(4));
        printf("» Performance log written to %s\n", GPT2_PERFORMANCE_JSON_FILE_NAME);
    } else{
        perror("Error opening output file");
        exit(EXIT_FAILURE);
    }
    
    json_decref(perf_root);
    //fclose(fp);

    return 1;
}