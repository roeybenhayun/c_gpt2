
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define a_ROW 64
#define a_COL 64
#define b_ROW 1
#define b_COL a_ROW
#define EPS (1e-6)


/// nvcc -o add_bias_2d_gpu add_bias_2d_gpu.c -lcudnn

float a_h[a_ROW][a_COL] = {};
float b_h[b_ROW][b_COL] = {};

float result_h1[a_ROW][a_COL] = {};
float result_h2[a_ROW][a_COL] = {};

// pointer to GPU RAM ///
float *a_d;
float *b_d;

cudnnTensorDescriptor_t a_desc, b_desc;

// if out is null addition is inplace 
static void add_bias_2d_cpu(float *a, int a_r, int a_c, float *b, float *out){
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

static void add_bias_2d_gpu(float *a, int a_r, int a_c, float *b, float *out){
    int a_dims[3]={0};
    int a_strides[3] = {0};
    int b_dims[3] = {0};
    int b_strides[3] = {0};
    a_dims[0] = 1;
    a_dims[1] = a_r;
    a_dims[2] = a_c;
    a_strides[0] = a_r * a_c;
    a_strides[1] = a_c;
    a_strides[2] = 1;

    b_dims[0] = 1;
    b_dims[1] = 1;
    b_dims[2] = a_c;
    b_strides[0] = a_r * a_c;
    b_strides[1] = a_c;
    b_strides[2] = 1;

    cudnnSetTensorNdDescriptor(a_desc,CUDNN_DATA_FLOAT,3,a_dims,a_strides);
    cudnnSetTensorNdDescriptor(b_desc,CUDNN_DATA_FLOAT,3,b_dims,b_strides);

}




int main(){

    cudnnHandle_t handle;
    
    float alpha = 1.0f;
    float beta = 1.0f;

    int a_dims[3] = {1,a_ROW,a_COL};
    int a_strides[3] = {a_ROW*a_COL,a_COL,1};

    int b_dims[3] = {1,b_ROW,b_COL};
    int b_strides[3] = {b_ROW*b_COL,b_COL,1};

    srand(time(NULL)); 

    // set a_h
    for(int i = 0; i < a_ROW ; i++){
        for (int j = 0; j < a_COL ; j++){
            a_h[i][j] = (float)rand()/RAND_MAX;
        }
    }
    // set b_h
    for(int i = 0; i < b_ROW ; i++){
        for (int j = 0; j < b_COL ; j++){
            b_h[i][j] = (float)rand()/RAND_MAX;
        }
    }
    
    cudnnCreate(&handle);
    cudnnCreateTensorDescriptor(&a_desc);
    cudnnCreateTensorDescriptor(&b_desc);
    
    // allocate a pointer
    cudnnSetTensorNdDescriptor(a_desc,CUDNN_DATA_FLOAT,3,a_dims,a_strides);
    cudnnSetTensorNdDescriptor(b_desc,CUDNN_DATA_FLOAT,3,b_dims,b_strides);

    cudaMalloc((void*)&a_d,a_ROW * a_COL * sizeof(float));
    cudaMemcpy(a_d,a_h,a_ROW * a_COL * sizeof(float),cudaMemcpyHostToDevice);

    cudaMalloc((void*)&b_d, b_ROW * b_COL * sizeof(float));
    cudaMemcpy(b_d,b_h,b_ROW * b_COL * sizeof(float),cudaMemcpyHostToDevice);


    // GPU addition
    // cudnn addition
    cudnnAddTensor(handle,&alpha,b_desc,b_d,&beta,a_desc,a_d);

    // copy back from GPU to CPU 
    cudaMemcpy(result_h1,a_d,a_ROW*a_COL*(sizeof(float)),cudaMemcpyDeviceToHost);


    printf("CUDNN addition has completed\n");

    add_bias_2d_cpu(&a_h[0][0],a_ROW,a_COL,&b_h[0][0],&result_h2[0][0]);


    // compare results
    int is_match = 1;

    for (int i = 0; i < a_ROW; i++){
        for (int j = 0; j < a_COL; j++){
            if (abs(result_h1[i][j] - result_h2[i][j]) > EPS){
                printf("CPU and GPU Addition do NOT match\n");
                is_match = 0;
                break;
            }
        }
    }

    if(is_match){
        printf("CPU and GPU Addition match\n");
    }



    cudaFree(a_d);
    cudaFree(b_d);
    cudnnDestroyTensorDescriptor(a_desc);
    cudnnDestroyTensorDescriptor(b_desc);
    cudnnDestroy(handle);
    return 0;

}