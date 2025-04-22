#include <math.h>
#include <stdio.h>
#include <time.h>
#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#define CBLAS_ROW_MAJOR CblasRowMajor
#define CBLAS_NO_TRANS CblasNoTrans
#endif

#define APPLY_ATTENTION_SCALING (1)
#define PI (3.14159265358979323846)

void print_2d_tensor(float *a, int a_r, int a_c);
float mean_(float *x, int len);
float variance_(float *x,int len, float mean);
void softmax_2d(float *a, int a_r, int a_c, float * c_out);
void dot_2d(float *a,int a_r, int a_c, float*b,int b_r,int b_c,float* c_out,int apply_attention_scaling );
void transpose_2d(float *a, int a_r, int a_c, float*b);
void layernorm_2d(float *a, int a_r, int a_c,float * ln_gamma, float * ln_beta,float * out, float epsilon);
float gelu(float x);
void gelu_2d(float *a,int a_c, int a_r, float *out);
void apply_casual_masking(float * a, int size);

// TODO 
// * Numerical stability - substract the row's max is the standard trick
// * Batch support - softmax_3d()
void softmax_2d(float *a, int a_r, int a_c, float * c_out){    
    for (int i=0; i<a_r; i++){
        float sum = 0.0;
        for (int j=0; j<a_c; j++){
            float temp_exp_res = exp(*(a + i*a_c + j));
            *(c_out + i*a_c +j) = temp_exp_res;
            sum += temp_exp_res;   
        }
        for (int j=0; j<a_c; j++){
            *(c_out + i*a_c +j) = *(c_out + i*a_c +j)/sum;
        }                
    }
}
// switched to float for higher precision
// calc matched python code
void dot_2d(float *a,int a_r, int a_c, float*b,int b_r,int b_c,float* c_out,int apply_attention_scaling ){
#ifdef USE_ACCELERATE

    // Use Accelerate's BLAS implementation
    float alpha = 1.0;
    float beta = 0.0;

    // Optional attention scaling
    if (apply_attention_scaling) {
        alpha = 1.0 / sqrt((float)a_c);
    }

    cblas_sgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                a_r, b_c, a_c,
                alpha,
                a, a_c,
                b, b_c,
                beta,
                c_out, b_c);
    //return 0.0; // optional: you can return a checksum like before
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

void apply_casual_masking(float * a, int size){
    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            a[i * size + j] = -INFINITY;
        }
    }
}
void print_2d_tensor(float *a, int a_r, int a_c)
{
    printf("[");
    for (int i=0; i<a_r; i++){
        printf("  [");
        for (int j=0; j<a_c; j++){
            printf(" %6.2f",*(a +i*a_c + j));
            if (j < a_c - 1) printf(", ");
        }
        printf("]");
        if (i < a_r - 1) printf(",\n");
        else printf("\n");
    }
    printf("]\n");
}


void transpose_2d(float *a, int a_r, int a_c, float*b){
// TODO input check
    for (int i=0; i<a_r; i++){
        for (int j=0; j<a_c; j++){
            *(b + j*a_r +i) = *(a + i*a_c + j);
        }
    }
}
// TODO - add the math expression
void layernorm_2d(float *a, int a_r, int a_c, 
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

void add_2d(float *a, int a_r, int a_c, float *b, float *out){
    for (int i=0; i<a_r; i++){
        for (int j=0; j<a_c; j++){
            *(out +i*a_c + j) = *(a +i*a_c + j) + *(b +i*a_c + j);
        }
    }
}

// if out is null addition is inplace 
void add_bias_2d(float *a, int a_r, int a_c, float *b, float *out){
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

float mean_(float *x, int len){
    float sum = 0.0;
    for (int i=0; i<len; i++){
        sum += *(x+i);
    }
    return (sum/(float)len);
}
float variance_(float *x,int len, float mean){
    float sum = 0.0;
    for (int i=0; i<len; i++){
        sum += pow(((*(x+i)) - mean),2);
    }
    return (sum/(float)len);
}

float gelu(float x){
    float term = sqrt(2.0/PI);
    return 0.5 * x * (1 + tanh (term * (x + 0.044715*pow(x,3))));
}

void gelu_2d(float *a,int a_c, int a_r, float *out){
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


// GPT2 small
const int d_model = 768;
const int ctx_len = 1024;
const float eps = 0.000005;
struct timespec start,end;
const int num_layers = 12;
// TODO: Tokenizer block
// TODO: Positional Embeddings
float embeddings[ctx_len][d_model] = {}; // for now post positional embeddings. This would go into layer norm

float X_norm[ctx_len][d_model] = {};
float X_norm2[ctx_len][d_model] = {};

float W1[d_model][d_model*4] = {};
float b1[d_model*4] = {};
float X1[ctx_len][d_model*4] = {};
float X1_out[ctx_len][d_model*4] = {};
float W2[d_model*4][d_model] = {};
float b2[d_model] = {};
float X2[ctx_len][d_model] = {};
float X2_out[ctx_len][d_model] = {};
float Xf_out[ctx_len][d_model] = {};

float W_q[d_model][d_model] = {}; // learnable
float W_k[d_model][d_model] = {}; // learnable
float W_v[d_model][d_model] = {}; // learnable
float Q[ctx_len][d_model] = {};
float K[ctx_len][d_model] = {};
float K_T[d_model][ctx_len] = {};
float V[ctx_len][d_model] = {};
float attention_scores[ctx_len][ctx_len] = {};
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



typedef struct{
    float * W_q;
    float * W_k;
    float * W_v;
    float * W1;
    float * W2;
    float * b1;
    float * b2;
    float *ln1_gamma;
    float *ln1_beta;
    float *ln2_gamma;
    float *ln2_beta;
}TransformerBlockParams;



void transformer_block(float *input,
                        TransformerBlockParams * tbp,
                        float * output){
    // Layer Norm 1
    layernorm_2d(input,ctx_len,d_model,tbp->ln1_gamma,tbp->ln1_beta, &X_norm[0][0],eps);
    printf("ln1_completed\n");

    // QKV
    dot_2d(&X_norm[0][0],ctx_len,d_model,tbp->W_q,d_model,d_model,&Q[0][0],!APPLY_ATTENTION_SCALING);
    dot_2d(&X_norm[0][0],ctx_len,d_model,tbp->W_k,d_model,d_model,&K[0][0],!APPLY_ATTENTION_SCALING);
    dot_2d(&X_norm[0][0],ctx_len,d_model,tbp->W_v,d_model,d_model,&V[0][0],!APPLY_ATTENTION_SCALING);
    printf("QKV completed\n");

    // Attention
    transpose_2d(&K[0][0], ctx_len,d_model , &K_T[0][0]);
    printf("attention completed\n");

    //print_2d_tensor(&K_T[0][0],d_model,ctx_len);
    dot_2d(&Q[0][0],ctx_len,d_model,&K_T[0][0],d_model,ctx_len,&attention_scores[0][0],APPLY_ATTENTION_SCALING);
    //print_2d_tensor(&attention_scores[0][0],ctx_len,ctx_len);
    
    // Casual masking
    apply_casual_masking(&attention_scores[0][0],ctx_len);
    printf("casual masking completed\n");

    // Softmax
    softmax_2d(&attention_scores[0][0], ctx_len,ctx_len,&attention_weights[0][0]);
    //print_2d_tensor(&attention_weights[0][0],ctx_len,ctx_len);
    printf("softmax completed\n");
    
    // Context
    dot_2d(&attention_weights[0][0],ctx_len,ctx_len,&V[0][0],ctx_len,d_model,&context[0][0],!APPLY_ATTENTION_SCALING);
    //print_2d_tensor(&context[0][0],ctx_len,d_model);
    printf("context completed\n");

    // Residuals
    add_2d(input,ctx_len,d_model,&context[0][0],&residual_out[0][0]);
    printf("residuals completed\n");

    // Layer Norm 2
    layernorm_2d(&residual_out[0][0],ctx_len,d_model,tbp->ln2_gamma,tbp->ln2_beta, &X_norm2[0][0],eps);
    printf("ln2 completed\n");

    // MLP layer 
    dot_2d(&X_norm2[0][0],ctx_len,d_model,tbp->W1,d_model,d_model*4,&X1_out[0][0],!APPLY_ATTENTION_SCALING);
    printf("mlp l1 w completed\n");
    //add_bias_2d();
    add_bias_2d(&X1_out[0][0],ctx_len,d_model*4,tbp->b1,NULL);
    printf("mlp l1 b completed\n");
    
    gelu_2d(&X1_out[0][0],ctx_len,d_model*4,NULL);
    printf("mlp gelu completed\n");

    dot_2d(&X1_out[0][0],ctx_len,d_model*4,tbp->W2,d_model*4,d_model,&X2_out[0][0],!APPLY_ATTENTION_SCALING);
    printf("mlp l2 w completed\n");

    add_bias_2d(&X2_out[0][0],ctx_len,d_model,tbp->b2,NULL);
    printf("mlp l2 b completed\n");

    add_2d(&X2_out[0][0],ctx_len,d_model,&residual_out[0][0],&residual2_out[0][0]);
    printf("mlp residuals completed\n");

    // last one - move out of the loop
    //layernorm_2d(&residual2_out[0][0],ctx_len,d_model,tbp->gamma,ln_f->beta,output,eps);
}

void load_layers_weights(TransformerBlockParams * p_tfb, int layer_id){
    // load layer weights
}

int main()
{
    clock_gettime(CLOCK_MONOTONIC, &start);
    printf("GPT2 Inference - Start\n");
    
    TransformerBlockParams layer = {
        &W_q[0][0],
        &W_k[0][0],
        &W_v[0][0],
        &W1[0][0],
        &W2[0][0],
        &b1[0],
        &b2[0],
        layer_norm1_gamma,
        layer_norm1_beta,
        layer_norm2_gamma,
        layer_norm2_beta
    };
    // Get user input 
    // TODO - Tokenizer (encode)

    for (int i=0 ; i < num_layers; i++){
        
        load_layers_weights(&layer, i);
        transformer_block(&embeddings[0][0],&layer, &embeddings[0][0]);
        printf("*****Layer %d completed******\n",i);
    }
    

    // argmax 

    // TODO - Tokenizer (decode?)

    clock_gettime(CLOCK_MONOTONIC, &end);
    float elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("GPT2 Inference - End\n");
    printf("Inference time =  %.2f seconds\n",elapsed);

    return 1;
}