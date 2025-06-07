#include <math.h>   // Not strictly needed for this code, but harmless
#include <stdio.h>
#include <time.h>   // For clock_gettime
#include <stdlib.h> // Good for matrix initialization with random values, or malloc if needed

// IMPORTANT: You'll need to include the CBLAS header for cblas_sgemm
// For OpenBLAS/ATLAS, it's typically:
//#include <cblas.h>
#include <Accelerate/Accelerate.h>
#define CBLAS_ROW_MAJOR CblasRowMajor
#define CBLAS_NO_TRANS CblasNoTrans

const int ctx_len = 1024;
// Global arrays are fine for this size, but for larger matrices
// dynamic allocation (malloc) is generally preferred to avoid stack overflow.
float a[ctx_len][ctx_len] = {0};
float b[ctx_len][ctx_len] = {0};
// It's good practice to have separate output matrices for each method
// to avoid one overwriting the results of the other if you want to verify correctness.
float c_manual[ctx_len][ctx_len] = {0}; // For your manual implementation
float c_cblas[ctx_len][ctx_len] = {0};  // For the CBLAS implementation

// Function to initialize matrices (Highly Recommended)
void initialize_matrices() {
    srand(time(NULL)); // Seed the random number generator once

    for (int i = 0; i < ctx_len; i++) {
        for (int j = 0; j < ctx_len; j++) {
            // Initialize with random float values between 0.0 and 1.0 (or -1.0 and 1.0)
            a[i][j] = (float)rand() / (float)RAND_MAX;
            b[i][j] = (float)rand() / (float)RAND_MAX;
        }
    }
}

// Function to verify results (Very Important for correctness check)
// This function assumes both matrices are in row-major order.
void verify_results() {
    float max_diff = 0.0f;
    for (int i = 0; i < ctx_len; i++) {
        for (int j = 0; j < ctx_len; j++) {
            float diff = fabsf(c_manual[i][j] - c_cblas[i][j]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }
    printf("Maximum difference between manual and CBLAS results: %e\n", max_diff);
    // A small non-zero difference is expected due to floating-point precision differences
    // in different optimized algorithms. A large difference indicates an error.
    if (max_diff > 1e-4) { // Example threshold, adjust as needed
        printf("WARNING: Results differ significantly! There might be an issue.\n");
    } else {
        printf("Results are consistent (within floating-point tolerance).\n");
    }
}


int main(){
    initialize_matrices(); // Initialize 'a' and 'b' with data

    struct timespec start,end;
    double elapsed_time; // Use double for elapsed time for better precision

    // --- Manual Dot Product ---
    printf("Starting manual dot_2d (CPU)...\n");
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < ctx_len; i++ ){
        for (int j = 0; j < ctx_len; j++ ){
            c_manual[i][j] = 0.0f; // Ensure initialization for each element
            for (int k = 0; k < ctx_len; k++){
                c_manual[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_time = (end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Manual dot_2d (CPU) finished in %.4f seconds\n", elapsed_time); // Increased precision

    // --- CBLAS Dot Product (SGEMM) ---
    printf("\nStarting CBLAS sgemm...\n");
    clock_gettime(CLOCK_MONOTONIC, &start);

    // cblas_sgemm parameters:
    // CblasRowMajor      - Specifies row-major storage order for all matrices. This is correct for your `float[ctx_len][ctx_len]` arrays.
    // CblasNoTrans       - 'a' is not transposed. Correct.
    // CblasNoTrans       - 'b' is not transposed. Correct.
    // M = ctx_len        - Rows of A (and C). Correct.
    // N = ctx_len        - Columns of B (and C). Correct.
    // K = ctx_len        - Inner dimension (columns of A / rows of B). Correct.
    // alpha = 1.0f       - Scalar multiplier for A*B. Correct.
    // A = a              - Pointer to matrix A. Correct.
    // lda = ctx_len      - Leading dimension of A. For row-major `A[M][K]`, lda is K. Here it's `ctx_len`. Correct.
    // B = b              - Pointer to matrix B. Correct.
    // ldb = ctx_len      - Leading dimension of B. For row-major `B[K][N]`, ldb is N. Here it's `ctx_len`. Correct.
    // beta = 0.0f        - Scalar multiplier for C. `0.0f` means `C = alpha * A*B + 0 * C`, so `C` is overwritten. This is what you want.
    // C = c_cblas        - Pointer to matrix C (where results are stored). Make sure this is `c_cblas` to avoid overwriting.
    // ldc = ctx_len      - Leading dimension of C. For row-major `C[M][N]`, ldc is N. Here it's `ctx_len`. Correct.

    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                ctx_len,
                ctx_len,
                ctx_len,
                1.0f,          // Use 1.0f for float literal
                (const float*)a, // Cast to const float* as per cblas_sgemm signature
                ctx_len,
                (const float*)b, // Cast to const float*
                ctx_len,
                0.0f,          // Use 0.0f for float literal
                (float*)c_cblas, // Cast to float*
                ctx_len
               );
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_time = (end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;
    printf("CBLAS sgemm finished in %.4f seconds\n", elapsed_time);

    // --- Verify Results ---
    verify_results();

    return 0;
}