/// TODO
// * Add non optimzed 2d dot product (mainly for performance comparison)
// * Change to loop to output token by token (instead of all at once)
// * Move the token selection logic into a separate function
// * Add KV cache
// * CLI arg for temperature
// * CLI arg for top_k
// * CLI atg for top_p
// * CLI arg for max tokens 
// * Function to cleanup data structures
// * calc time to token output
// * Update layer names consistently

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

#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#define CBLAS_ROW_MAJOR CblasRowMajor
#define CBLAS_NO_TRANS CblasNoTrans
#endif

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

static void dot_2d(float *a,int a_r, int a_c, int lda, float*b,int b_r,int b_c,int ldb, float* c_out,int c_r, int c_c, int ldc, int transpose_b,int apply_attention_scaling ){
#ifdef USE_ACCELERATE
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

#else    
    float dot_product = 0.0;
    float dot_product_sum = 0.0;
    float scale_factor = 1.0;
    // TODO - check edge cases here
    if(apply_attention_scaling == 1){
        scale_factor = (float)(1.0/sqrt(a_c));
    }
    //printf("in dot_2d\n");
    for (int i=0; i<a_r; i++){        
        for (int j=0; j<b_c; j++){            
            for (int k=0; k<b_r; k++){
                //printf("%d,%d,  %d,%d\n",i,k,k,j);
                // dot_product += a[i][k] * b[k][j];
                float av = *(a + i * a_c + k);
                float bv = *(b + k * b_c + j);
                dot_product += av * bv;     
            }
            dot_product_sum += dot_product;
            *(c_out + i*b_c +j) = dot_product * scale_factor;
            // Casual mask - take outside of the loop
            //if (j > i){ // future token, mask it
            //    *(c_out + i*b_c +j) = -INFINITY;
            //}
            //dot_product = 0.0;
        }
    }
    //printf("dot_2d_product = %.17f\n",dot_product_sum);
    //return dot_product_sum;
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

// GPT2 small
const int d_model = 768;
const int ctx_len = 1024;
const float eps = 0.00001;
const int num_layers = 12;
const int vocab_size = 50257;
const int nof_heads = 12;
const int head_dim = d_model/nof_heads;
const int d_ff = d_model * 4;

float wte[vocab_size][d_model] = {};
float wte_T[d_model][vocab_size] = {};

float wpe[ctx_len][d_model] = {};
float embeddings[ctx_len][d_model] = {}; // for now post positional embeddings. This would go into layer norm

float X_norm[ctx_len][d_model] = {};
float X_norm2[ctx_len][d_model] = {};

float W1[d_ff][d_model] = {};
float b1[d_ff] = {};
float X1[ctx_len][d_ff] = {};
float X1_out[ctx_len][d_ff] = {};
float W2[d_model][d_ff] = {};
float b2[d_model] = {};
float X2[ctx_len][d_model] = {};
float X2_out[ctx_len][d_model] = {};
float Xf_out[ctx_len][d_model] = {};

float W_q[d_model][d_model] = {}; // learnable
float W_k[d_model][d_model] = {}; // learnable
float W_v[d_model][d_model] = {}; // learnable
float b_q[d_model] = {}; // learnable
float b_k[d_model] = {}; // learnable
float b_v[d_model] = {}; // learnable

float temp_attn_weight[3*d_model][d_model]; // 2304 = 768 * 3
float temp_attn_bias[3*d_model];

float Q[ctx_len][d_model] = {};
float K[ctx_len][d_model] = {};
float K_T[d_model][ctx_len] = {};
float V[ctx_len][d_model] = {};
float attention_scores[ctx_len][ctx_len] = {};
float attention_scores_temp[ctx_len][ctx_len] = {};
float attention_weights[ctx_len][ctx_len] = {};
float context[ctx_len][d_model] = {};

float layer_norm1_gamma[d_model] = {}; // default: no scaling
float layer_norm1_beta[d_model] = {};  // default: no shifting
float layer_norm2_gamma[d_model] = {}; // default: no scaling
float layer_norm2_beta[d_model] = {};  // default: no shifting
float layer_normf_gamma[d_model] = {}; // default: no scaling
float layer_normf_beta[d_model] = {};  // default: no shifting

float residual_out[ctx_len][d_model] = {};
float residual2_out[ctx_len][d_model] = {};

float attn_proj_weight[d_model][d_model] = {};
float attn_proj_bias[d_model] = {};

float logits[ctx_len][vocab_size] = {};

float context_heads[nof_heads][ctx_len][head_dim];
float scores_h[ctx_len][ctx_len];
float weights_h[ctx_len][ctx_len];
float final_attention_output[ctx_len][d_model]; 

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


int glob_idx = 0;
static void transformer_block(float *input,int n_tokens,
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


    // QKV
    dot_2d(&X_norm[0][0],n_tokens,d_model,d_model,tbp->W_q,d_model,d_model,d_model,&Q[0][0],n_tokens, d_model,d_model,1,!APPLY_ATTENTION_SCALING);
    add_bias_2d(&Q[0][0],n_tokens,d_model,tbp->b_q,NULL);
    
    // save json
    //add_tensor_to_layer(layer_obj, "Q", &Q[0][0], ctx_len, d_model, n_tokens, d_model);

    dot_2d(&X_norm[0][0],n_tokens,d_model,d_model,tbp->W_k,d_model,d_model,d_model,&K[0][0],n_tokens,d_model,d_model,1,!APPLY_ATTENTION_SCALING);
    add_bias_2d(&K[0][0],n_tokens,d_model,tbp->b_k,NULL);
    // save json
    //add_tensor_to_layer(layer_obj, "K", &K[0][0], ctx_len, d_model, n_tokens, d_model);

    dot_2d(&X_norm[0][0],n_tokens,d_model,d_model,tbp->W_v,d_model,d_model,d_model,&V[0][0],n_tokens,d_model,d_model,1,!APPLY_ATTENTION_SCALING);
    add_bias_2d(&V[0][0],n_tokens,d_model,tbp->b_v,NULL);
    //add_tensor_to_layer(layer_obj, "V", &V[0][0], ctx_len, d_model, n_tokens, d_model);
    print_2d_tensor("C Q[1][:10]:",&Q[1][0],ctx_len,d_model,1,10,0);
    print_2d_tensor("C K[1][:10]:",&K[1][0],ctx_len,d_model,1,10,0);
    print_2d_tensor("C V[1][:10]:",&V[1][0],ctx_len,d_model,1,10,0);


    // ** to add here the multi head attention code ***
    for (int h=0 ; h < nof_heads; h++){
        float *q_h = &Q[0][0]+ h * head_dim;
        float *k_h = &K[0][0]+ h * head_dim;
        float *v_h = &V[0][0]+ h * head_dim;
        float *context_h_out = &context_heads[h][0][0];

        //Clear scores buffer to avoid stale values
        memset(scores_h, 0, sizeof(float) * ctx_len * ctx_len);

        dot_2d(q_h,n_tokens,head_dim,d_model,k_h,n_tokens,head_dim,d_model,&scores_h[0][0],n_tokens,n_tokens,ctx_len,1,APPLY_ATTENTION_SCALING);
        //print_2d_tensor(&scores_h[0][0], 4, 4, 4, 4);
        apply_casual_masking(&scores_h[0][0],ctx_len/*n_tokens*/,n_tokens);
        print_2d_tensor("C scores_h[1][i] (before Softmax):",&scores_h[1][0],ctx_len,ctx_len,1,10,0);

        
        softmax_2d(&scores_h[0][0], n_tokens,n_tokens,ctx_len, &weights_h[0][0]);        
        print_2d_tensor("C weights_h[1][:10]",&weights_h[1][0],ctx_len,ctx_len,1,10,0);

        dot_2d(&weights_h[0][0],n_tokens,n_tokens,ctx_len,v_h,n_tokens,head_dim,d_model,context_h_out,n_tokens,head_dim,head_dim,0,!APPLY_ATTENTION_SCALING);


    }

    // use memcpy instead the loop
    for (int i = 0; i < n_tokens; i++) {
        for (int h = 0; h < nof_heads; h++) {
            for (int k = 0; k < head_dim; k++) {
                 // Copy element from context_heads[i][h][k] to final_attention_output[i][h * head_dim + k]
                 float source_value = context_heads[h][i][k]; // Correct index for [nof_heads][ctx_len][head_dim] layout
                 int dest_col = (h * head_dim) + k;
                 final_attention_output[i][dest_col] = source_value;
            }
        }
    }
    print_2d_tensor("C final_attention_output[1][:10]:",&final_attention_output[1][0],ctx_len,d_model,1,10,0);

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

    //memcpy(residual2_out, residual_out, sizeof(residual_out));
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

int main()
{
    const int n_tokens = 1024;
    char input_buffer[2048];    
    char encode_request[2048];
    char encode_response[2048];
    char decode_request[2048];
    char decode_response[2048];
    int tokens[n_tokens] = {0};
    char token_list[8192] = {0};
    

    long offset = (vocab_size * d_model + ctx_len * d_model) * sizeof(float);
    char temp[32];
    

    TransformerBlockParams layer = {
        &W_q[0][0],
        &W_k[0][0],
        &W_v[0][0],
        &b_q[0],
        &b_k[0],
        &b_v[0],
        &attn_proj_weight[0][0],
        &attn_proj_bias[0],
        &W1[0][0],
        &W2[0][0],
        &b1[0],
        &b2[0],
        layer_norm1_gamma,
        layer_norm1_beta,
        layer_norm2_gamma,
        layer_norm2_beta
    };
    // Load GTP2 weights
    printf("Loading GPT2 weights...\n");
    
    // Open the file containing the weights
    FILE * fp = fopen("gpt2_weights.bin","rb");
    
    json_t *json_root = json_object();
    
    fread_or_exit(wte,sizeof(float),vocab_size*d_model,fp);
    fread_or_exit(wpe,sizeof(float),ctx_len*d_model,fp);
    transpose_2d(&wte[0][0], vocab_size,d_model , &wte_T[0][0]);


    float temperature = 0.9;
    int max_out_tokens = 30;

    while(1){ 
        
        printf("Enter Input:");
        // Input
        fgets(input_buffer, sizeof(input_buffer), stdin);
        input_buffer[strcspn(input_buffer, "\n")] = 0;
        if (strcmp(input_buffer, "q") == 0) {
            printf("QUIT\n");            
            break;
        }

        clock_gettime(CLOCK_MONOTONIC, &start);
        printf("GPT2 Inference - Start\n");

        // cleanup, move to the end (move to function)
        memset(tokens,0,sizeof(tokens));
        memset(embeddings,0,sizeof(embeddings));            
        memset(encode_request, 0, sizeof(encode_request));
        memset(encode_response, 0, sizeof(encode_response));
        memset(decode_request, 0, sizeof(decode_request));
        memset(decode_response, 0, sizeof(decode_response));
        memset(token_list,0,sizeof(token_list));

        // Format
        snprintf(encode_request, sizeof(encode_request),
         "{\"mode\": \"encode\", \"text\": \"%s\"}", input_buffer);
         // Get tokens
        send_json_to_tokenizer(encode_request, encode_response);
        printf("Encode Response: %s\n", encode_response);
        // Format
        int n_tokens = parse_tokens(encode_response, tokens, ctx_len);
        if (n_tokens < 0) {
            printf("Failed to parse tokens!\n");
            continue;
        }
        printf("Parsed %d tokens\n",n_tokens);
        
        int last_token_position = n_tokens - 1; 
        int ii = 0;
        for (; ii < max_out_tokens; ii++){  

            // --- Reset top_k arrays ---
            int top_k_indices[40];
            float top_k_values[40];
            for (int j = 0; j < 40; j++) {
                top_k_indices[j] = -1;
                top_k_values[j] = -INFINITY;
            }
        
            // Set
            for (int i=0; i<n_tokens;i++){
                for (int j=0; j<d_model;j++){
                    // can optimize here
                    embeddings[i][j] = wte[tokens[i]][j] + wpe[i][j];
                }
            }

            // reset file pointer position to load weights from the right position
            fseek(fp, offset, SEEK_SET);
            // Infer
            for (int i=0 ; i < num_layers; i++){
                //printf("File position before layer %d = %ld\n", i, ftell(fp));
                load_layers_weights(&layer, i,fp);
                //printf("nof_tokens=%d\n",n_tokens);
                                                
                transformer_block(&embeddings[0][0],n_tokens,&layer,json_root,i,ii);
                
                // Zero out entire embeddings buffer (safe reset)
                memset(&embeddings[0][0], 0, sizeof(float) * ctx_len * d_model);
                // use the output from this block as the input to the next block
                memcpy(&embeddings[0][0], &residual2_out[0][0], sizeof(float) * n_tokens * d_model);
                //printf("*****Layer %d completed******\n",i);

            }

            // load weights and bias here
            fread_or_exit(layer_normf_gamma, sizeof(float), d_model, fp);
            fread_or_exit(layer_normf_beta, sizeof(float), d_model, fp);
            

            print_2d_tensor("C residual2_out[1][:10]:",&residual2_out[1][0],ctx_len,d_model,1,10,0);

            // Layer Norm final
            layernorm_2d(&residual2_out[0][0],n_tokens,d_model,&layer_normf_gamma[0],&layer_normf_beta[0],&Xf_out[0][0],eps);
            print_2d_tensor("C Xf_out[1][:10]:",&Xf_out[1][0],ctx_len,d_model,1,10,0);

            // get logits
            dot_2d(&Xf_out[0][0],n_tokens,d_model,d_model,&wte_T[0][0],d_model,vocab_size,vocab_size,&logits[0][0],n_tokens,vocab_size,vocab_size,0,!APPLY_ATTENTION_SCALING);
            print_2d_tensor("C logits[1][i]",&logits[1][0],ctx_len,vocab_size,1,10,0);


        
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

            // Sample from probability distribution
            float r = (float)rand() / RAND_MAX;
            float cumulative = 0.0;
            int sampled_token = -1;
            for (int i = 0; i < vocab_size; i++) {
                cumulative += probs[i];
                if (r < cumulative) {
                    sampled_token = i;
                    break;
                }
            }


            //// Use sampled_token as your predicted next token
            tokens[n_tokens++] = sampled_token;
            last_token_position++;

                    // Find top 10 tokens by probability
            int top_indices[10] = {0};
            float top_probs[10] = {-1};

            for (int i = 0; i < vocab_size; i++) {
                for (int j = 0; j < 10; j++) {
                    if (probs[i] > top_probs[j]) {
                        // Shift down
                        for (int k = 9; k > j; k--) {
                            top_probs[k] = top_probs[k - 1];
                            top_indices[k] = top_indices[k - 1];
                        }
                        // Insert
                        top_probs[j] = probs[i];
                        top_indices[j] = i;
                        break;
                    }
                }
            }

            printf("Top 10 predicted tokens:\n");
            for (int i = 0; i < 10; i++) {
                printf("Token %d: Prob = %.4f\n", top_indices[i], top_probs[i]);
            }

        }
    
        for (int i = 0; i < n_tokens; i++) {
            printf("*******TOKEN = %d *****\n",tokens[i]);
            snprintf(temp, sizeof(temp), "%d%s", tokens[i], (i < n_tokens - 1) ? "," : "");
            strcat(token_list, temp);
        }

        // Now format the decode_request
        snprintf(decode_request, sizeof(decode_request),
        "{\"mode\": \"decode\", \"tokens\": [%s]}", token_list);


        send_json_to_tokenizer(decode_request, decode_response);
        printf("Decode Response: %s\n", decode_response);


        printf("*****Final layer norm completed\n");

        clock_gettime(CLOCK_MONOTONIC, &end);
        float elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("GPT2 Inference - End\n");
        printf("Inference time =  %.2f seconds\n",elapsed);
    
    }//main loop

    
    FILE *json_out = fopen("logs/tensor_output.json", "w");
    json_dumpf(json_root, json_out, JSON_INDENT(4));
    fclose(json_out);
    json_decref(json_root);

    fclose(fp);

    return 1;
}