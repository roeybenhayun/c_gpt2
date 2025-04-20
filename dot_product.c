#include <stdio.h>


// switched to double for higher precision
// calc matched python code
float dot_2d(double *a,int a_r, int a_c, double*b,int b_r,int b_c,double* c_out){
    float dot_product = 0.0;
    float dot_product_sum = 0.0;
    printf("in dot_2d\n");
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
            *(c_out + i*b_c +j) = dot_product;
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
int main()
{
    double c[3][4] = {
        {1.2,   2.3,    3,  4.9 },
        {1.2,   2.3,    3,  4   },
        {1.2,   2.3,    3,  4.9 }
    }; //3x4
    double d[4][3] = {
        {0.3,   0.9,    1.1 },
        {0.4,   0.9,    1.1 },
        {0.5,   0.9,    1.1 },
        {0.5,   0.9,    1.1 }
    }; //4x3


    double e_out[3][3] = {};

    print_2d_tensor(&c[0][0],3,4);
    print_2d_tensor(&d[0][0],4,3);

    dot_2d(&c[0][0],3,4,&d[0][0],4,3,&e_out[0][0]);

    print_2d_tensor(&e_out[0][0],3,3);

    return 1;
}