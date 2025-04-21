#include <math.h>
#include <stdio.h>
#include <time.h>

#define APPLY_ATTENTION_SCALING (1)
#define PI (3.14159265358979323846)

void print_2d_tensor(double *a, int a_r, int a_c);
double mean_(double *x, int len);
double variance_(double *x,int len, double mean);
void softmax_2d(double *a, int a_r, int a_c, double * c_out);
double dot_2d(double *a,int a_r, int a_c, double*b,int b_r,int b_c,double* c_out,int apply_attention_scaling );
void transpose_2d(double *a, int a_r, int a_c, double*b);
void layernorm_2d(double *a, int a_r, int a_c,double * ln_gamma, double * ln_beta,double * out, double epsilon);
double gelu(double x);
void gelu_2d(double *a,int a_c, int a_r, double *out);


// TODO 
// * Numerical stability - substract the row's max is the standard trick
// * Batch support - softmax_3d()
void softmax_2d(double *a, int a_r, int a_c, double * c_out){    
    for (int i=0; i<a_r; i++){
        double sum = 0.0;
        for (int j=0; j<a_c; j++){
            double temp_exp_res = exp(*(a + i*a_c + j));
            *(c_out + i*a_c +j) = temp_exp_res;
            sum += temp_exp_res;   
        }
        for (int j=0; j<a_c; j++){
            *(c_out + i*a_c +j) = *(c_out + i*a_c +j)/sum;
        }                
    }
}
// switched to double for higher precision
// calc matched python code
double dot_2d(double *a,int a_r, int a_c, double*b,int b_r,int b_c,double* c_out,int apply_attention_scaling ){
    double dot_product = 0.0;
    double dot_product_sum = 0.0;
    double scale_factor = 1.0;
    // TODO - check edge cases here
    if(apply_attention_scaling == 1){
        scale_factor = (double)(1.0/sqrt(a_c));
    }
    //printf("in dot_2d\n");
    for (int i=0; i<a_r; i++){        
        for (int j=0; j<b_c; j++){            
            for (int k=0; k<b_r; k++){
                //printf("%d,%d,  %d,%d\n",i,k,k,j);
                // dot_product += a[i][k] * b[k][j];
                double av = *(a + i * a_c + k);
                double bv = *(b + k * b_c + j);
                dot_product += av * bv;     
            }
            dot_product_sum += dot_product;
            *(c_out + i*b_c +j) = dot_product * scale_factor;
            // Casual mask
            if (j > i){ // future token, mask it
                *(c_out + i*b_c +j) = -INFINITY;
            }
            dot_product = 0.0;
        }
    }
    //printf("dot_2d_product = %.17f\n",dot_product_sum);
    return dot_product_sum;
}

void print_2d_tensor(double *a, int a_r, int a_c)
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


void transpose_2d(double *a, int a_r, int a_c, double*b){
// TODO input check
    for (int i=0; i<a_r; i++){
        for (int j=0; j<a_c; j++){
            *(b + j*a_r +i) = *(a + i*a_c + j);
        }
    }
}
// TODO - add the math expression
void layernorm_2d(double *a, int a_r, int a_c, 
                    double * ln_gamma, double * ln_beta,
                    double * out, double epsilon){
    
    for (int i=0; i < a_r; i++){
        double *row = a + i * a_c;
        double mean = mean_(row,a_c);
        double var = variance_(row,a_c,mean);
        for (int j=0; j<a_c; j++){
            *(out + i*a_c + j) = *(ln_gamma+j) * ((*(a + i*a_c + j) - mean)/sqrt(var+epsilon)) + *(ln_beta+j);
        }
    }
}

void add_2d(double *a, int a_r, int a_c, double *b, double *out){
    for (int i=0; i<a_r; i++){
        for (int j=0; j<a_c; j++){
            *(out +i*a_c + j) = *(a +i*a_c + j) + *(b +i*a_c + j);
        }
    }
}

// if out is null addition is inplace 
void add_bias_2d(double *a, int a_r, int a_c, double *b, double *out){
    double * tmp = out;
    if (out == NULL){ //inplace
        tmp = a;
    } 
    for (int i=0; i<a_r; i++){
        for (int j=0; j<a_c; j++){
            *(tmp +i*a_c + j) = *(a +i*a_c + j) + *(b + j);
        }
    }
}

double mean_(double *x, int len){
    double sum = 0.0;
    for (int i=0; i<len; i++){
        sum += *(x+i);
    }
    return (sum/(double)len);
}
double variance_(double *x,int len, double mean){
    double sum = 0.0;
    for (int i=0; i<len; i++){
        sum += pow(((*(x+i)) - mean),2);
    }
    return (sum/(double)len);
}

double gelu(double x){
    double term = sqrt(2.0/PI);
    return 0.5 * x * (1 + tanh (term * (x + 0.044715*pow(x,3))));
}

void gelu_2d(double *a,int a_c, int a_r, double *out){
    double * tmp = out;
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
const double eps = 0.000005;
struct timespec start,end;
const int num_layers = 12;
// TODO: Tokenizer block
// TODO: Positional Embeddings
double embeddings[ctx_len][d_model] = {}; // for now post positional embeddings. This would go into layer norm

double X_norm[ctx_len][d_model] = {};
double X_norm2[ctx_len][d_model] = {};

double W1[d_model][d_model*4] = {};
double b1[d_model*4] = {};
double X1[ctx_len][d_model*4] = {};
double X1_out[ctx_len][d_model*4] = {};
double W2[d_model*4][d_model] = {};
double b2[d_model] = {};
double X2[ctx_len][d_model] = {};
double X2_out[ctx_len][d_model] = {};
double Xf_out[ctx_len][d_model] = {};

double W_q[d_model][d_model] = {}; // learnable
double W_k[d_model][d_model] = {}; // learnable
double W_v[d_model][d_model] = {}; // learnable
double Q[ctx_len][d_model] = {};
double K[ctx_len][d_model] = {};
double K_T[d_model][ctx_len] = {};
double V[ctx_len][d_model] = {};
double attention_scores[ctx_len][ctx_len] = {};
double attention_weights[ctx_len][ctx_len] = {};
double context[ctx_len][d_model] = {};

double layer_norm1_gamma[d_model] = {}; // default: no scaling
double layer_norm1_beta[d_model] = {};  // default: no shifting
double layer_norm2_gamma[d_model] = {}; // default: no scaling
double layer_norm2_beta[d_model] = {};  // default: no shifting
double layer_normf_gamma[d_model] = {}; // default: no scaling
double layer_normf_beta[d_model] = {};  // default: no shifting

double residual_out[ctx_len][d_model] = {};
double residual2_out[ctx_len][d_model] = {};



typedef struct{
    double * W_q;
    double * W_k;
    double * W_v;
    double * W1;
    double * W2;
    double * b1;
    double * b2;
    double *ln1_gamma;
    double *ln1_beta;
    double *ln2_gamma;
    double *ln2_beta;
}TransformerBlockParams;



void transformer_block(double *input,
                        TransformerBlockParams * tbp,
                        double * output){
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
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("GPT2 Inference - End\n");
    printf("Inference time =  %.2f seconds\n",elapsed);

    return 1;
}