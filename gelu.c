
#include <math.h>
#include <stdio.h>
#define PI 3.14159265358979323846

double gelu(double x){
    double term = sqrt(2.0/PI);
    return 0.5 * x * (1 + tanh (term * (x + 0.044715*pow(x,3))));
}

void gelu_2d(double *a,int a_c, int a_r, double *out){
    for (int i=0; i<a_r; i++){
        for (int j=0; j<a_c; j++){
            *(out +i*a_c + j) = gelu(*(a +i*a_c + j));
        }
    }
}

int main()
{
    double in = 1.23;
    double val1 = gelu(in);
    printf("GELU of %f = %6.2f\n",in,val1);
    return 1;
}