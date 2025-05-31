#include <math.h>
#include <stdio.h>


void softmax_1d(double*a,int len){
    double sum = 0.0;

    for(int i=0; i<len; i++){
        sum += exp(a[i]);
    }
    for(int i=0; i<len; i++){
        printf("softmax[%d] = %.17f\n",i, exp(a[i])/sum);
    }
}

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

int main()
{
    double a[4]={1.2,   2.3,    3,  4.9 };
    double b[4]={0.4,   2.3,    3,  4.9 };
    double c[4]={0,     0,      0,  0   };

    softmax_1d(a,4);
    softmax_1d(b,4);
    return 1;
}
