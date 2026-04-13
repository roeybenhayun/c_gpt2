#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>


/*** 
 * How to run this code 
 * nvcc gelu.cu -o gelu
 * ./caual_masking
 * ***/


//#define ctx_len (1024) 
//#define  d_model (768)
#define nof_tokens (6)
#define ctx_len (10) //rows  
#define  d_model (10) // columns

#define PI (3.14159265358979323846)

const float eps = 0.00001;
float input[ctx_len][d_model] = {};
float output[ctx_len][d_model] = {};
float output_gpu[ctx_len][d_model] = {};

float (*input_d)[d_model];
float (*output_d)[d_model];


void init_data(){
    int i=0,j=0;
    for (;i<ctx_len;i++){
        for (;j<d_model;j++){        
            input[i][j] = (float)rand()/RAND_MAX;
        }
        j=0;
    }
    memset(output,0,ctx_len*d_model*sizeof(float));
    memset(output_gpu,0,ctx_len*d_model*sizeof(float));
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

__global__ void gelu_kernel(float (*in)[d_model], int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    //float term = sqrt(2.0/PI);
    float term = 0.79788456f; // this is fixed value. move it to defines 
     
    if (row < rows && col < cols) {
        float val = in[row][col];        
        //in[row][col] =  0.5f * val * (1.0f + tanhf (term * (val + 0.044715f*powf(val,3.0f))));
        in[row][col] =  0.5f * val * (1.0f + tanhf (term * (val + 0.044715f*val*val*val)));
    }
}

void calc_gelu_gpu(){
    
    
    dim3 threadsPerBlock;
    threadsPerBlock.x = 32; 
    threadsPerBlock.y = 32;
    threadsPerBlock.z = 1;
    
    
    dim3 numBlocks;
    numBlocks.x = ((d_model + threadsPerBlock.x - 1) / threadsPerBlock.x);
    numBlocks.y = ((ctx_len + threadsPerBlock.y - 1) / threadsPerBlock.y);
    numBlocks.z = 1;


    printf("Launching Kernel with Grid(%d, %d) and Block(%d, %d)\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    // 3. Launch Kernel
    gelu_kernel<<<numBlocks, threadsPerBlock>>>(input_d,ctx_len,d_model);
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

}

void print_array(float * in, int in_r, int in_c){
    int i,j;
    //int j = 0;
    for (i = 0;i<in_r;i++){
        for (j =0;j<in_c;j++){            
            printf("[%d][%d]: %f\n",i,j,*(in + i*in_c + j));
        }
        j = 0;
        printf("\n");
    }
}
void init_gpu(){
    cudaError_t err;    
    err = cudaMalloc((void**)&input_d, d_model*ctx_len * sizeof (*input_d));
    if(err != cudaSuccess) printf("Alloc Error: %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void**)&output_d, d_model * ctx_len * sizeof(*output_d));
    if(err != cudaSuccess) printf("Alloc Error: %s\n", cudaGetErrorString(err));

}
void copy_to_gpu(){
    cudaMemcpy(input_d,  input, ctx_len * sizeof (*input_d),cudaMemcpyHostToDevice);
    cudaMemcpy(output_d,  output, ctx_len * sizeof(*output),cudaMemcpyHostToDevice);
}

void verify_results() {
    // Copy result back from GPU to CPU to compare
    cudaMemcpy(output_gpu, output_d, ctx_len * sizeof(*output_d), cudaMemcpyDeviceToHost);

    int errors = 0;
    for(int i = 0; i < ctx_len; i++) {
        for(int j = 0; j < d_model; j++) {
            float diff = fabs(output[i][j] - output_gpu[i][j]);
            if(diff > 1e-4) { // Allow small floating point error
                if(errors < 5) printf("Mismatch at [%d][%d]: CPU %f vs GPU %f\n", i, j, output[i][j], output_gpu[i][j]);
                errors++;
            }
        }
    }

    if(errors == 0) printf("SUCCESS: CPU and GPU results match!\n");
    else printf("FAILURE: Found %d mismatches.\n", errors);
}

int main(){
    printf("Initializing data\n");
    init_data();
    printf("Initializing GPU\n");
    init_gpu();

    printf("Copy to GPU...\n");
    copy_to_gpu();

    printf("Computing CPU...\n");
    gelu_2d(&input[0][0],ctx_len, d_model, NULL);
  
    printf("*****CPU RESULTS START\n");
    print_array(&input[0][0],ctx_len,d_model);
    printf("*****CPU RESULTS END\n");

    printf("Computing GPU...\n");
    calc_gelu_gpu();

    printf("*****GPU RESULTS START\n");
    print_array(&output_d[0][0],ctx_len,d_model);
    printf("*****GPU RESULTS END\n");

    //printf("Verifying...\n");
    verify_results();

    // Free GPU memory

    cudaFree(input_d);
    cudaFree(output_d);
    
}


