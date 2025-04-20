#include <math.h>
#include <stdio.h>

#define APPLY_ATTENTION_SCALING (1)
void print_2d_tensor(double *a, int a_r, int a_c);
double mean_(double *x, int len);
double variance_(double *x,int len, double mean);
void softmax_2d(double *a, int a_r, int a_c, double * c_out);
double dot_2d(double *a,int a_r, int a_c, double*b,int b_r,int b_c,double* c_out,int apply_attention_scaling );
void transpose_2d(double *a, int a_r, int a_c, double*b);
void layernorm_2d(double *a, int a_r, int a_c,double * ln_gamma, double * ln_beta,double * out, double epsilon);

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
    printf("dot_2d_product = %.17f\n",dot_product_sum);
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



int main()
{
    double attention_scores[3][3] = {};
    double attention_weights[3][3] = {};
    double context[3][4] = {};
    double K_T[4][3] = {};
    double layer_norm1_gamma[3] = {1,1,1}; // default: no scaling
    double layer_norm2_beta[3] = {0,0,0};  // default: no shifting

    double Q[3][4] = {
        {1.2,   2.3,    3,  4.9 },
        {1.2,   2.3,    3,  4   },
        {1.2,   2.3,    3,  4.9 }
    }; //3x4
    double K[3][4] = {
        {0.3,   0.9,    1.1,    1.2 },
        {0.4,   0.9,    1.1,    1.3 },
        {0.5,   0.9,    1.1,    1.5 }
    }; //3x4

    double V[3][4] = {
        {0.3,   0.9,    1.1,    0.4 },
        {0.4,   0.9,    1.1,    1.2 },
        {0.5,   0.9,    1.1,    0.1 }        
    }; //3x4

    //layernorm_2d()...
    transpose_2d(&K[0][0], 3,4 , &K_T[0][0]);
    print_2d_tensor(&K_T[0][0],4,3);

    dot_2d(&Q[0][0],3,4,&K_T[0][0],4,3,&attention_scores[0][0],APPLY_ATTENTION_SCALING);
    print_2d_tensor(&attention_scores[0][0],3,3);
    
    softmax_2d(&attention_scores[0][0], 3,3,&attention_weights[0][0]);
    print_2d_tensor(&attention_weights[0][0],3,3);
    
    dot_2d(&attention_weights[0][0],3,3,&V[0][0],3,4,&context[0][0],!APPLY_ATTENTION_SCALING);
    print_2d_tensor(&context[0][0],3,4);

    //add_2d()...

    return 1;
}