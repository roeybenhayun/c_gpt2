#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

// x86_64 (CPU)
#include <cblas.h>

#define EPS (1e-6)
#define WARMUP_ITERS 3
#define BENCH_ITERS  5

static cublasHandle_t get_cublas_handle() {
    static bool handle_initialized = false;
    static cublasHandle_t handle;
    if (!handle_initialized) {
        cublasStatus_t stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "FATAL: cuBLAS handle initialization failed\n");
            exit(1);
        }
        handle_initialized = true;
    }
    return handle;
}

static void dot_2d_cpu(float *a, int a_r, int a_c, int lda,
                       float *b, int b_r, int b_c, int ldb,
                       float *c_out, int c_r, int c_c, int ldc,
                       int transpose_b, int apply_attention_scaling) {
    float alpha = 1.0f;
    if (apply_attention_scaling)
        alpha = 1.0f / sqrtf((float)a_c);
    float beta = 0.0f;

    enum CBLAS_TRANSPOSE trans_b = transpose_b ? CblasTrans : CblasNoTrans;
    int M = a_r;
    int K = a_c;
    int N = transpose_b ? b_r : b_c;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, trans_b,
                M, N, K, alpha, a, lda, b, ldb, beta, c_out, ldc);
}

static void dot_2d_gpu(float *a, int a_r, int a_c, int lda,
                       float *b, int b_r, int b_c, int ldb,
                       float *c_out, int c_r, int c_c, int ldc,
                       int transpose_b, int apply_attention_scaling) {
    cublasHandle_t handle = get_cublas_handle();

    float alpha = 1.0f;
    if (apply_attention_scaling)
        alpha = 1.0f / sqrtf((float)a_c);
    const float beta = 0.0f;

    const int M = a_r;
    const int K = a_c;
    const int N = transpose_b ? b_r : b_c;

    const cublasOperation_t opB = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t opA = CUBLAS_OP_N;

    cublasStatus_t stat = cublasSgemm(handle, opB, opA, N, M, K, &alpha,
                                      b, ldb, a, lda, &beta, c_out, ldc);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "FATAL: cublasSgemm failed\n");
        exit(1);
    }
}

static double get_elapsed_sec(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) + (double)(end->tv_nsec - start->tv_nsec) / 1e9;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matrix_size> [--json]\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    if (N <= 0 || N > 32768) {
        fprintf(stderr, "Matrix size must be between 1 and 32768\n");
        return 1;
    }

    bool json_output = false;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0)
            json_output = true;
    }

    size_t mat_bytes = (size_t)N * N * sizeof(float);

    // Host memory
    float *a_h = (float *)malloc(mat_bytes);
    float *b_h = (float *)malloc(mat_bytes);
    float *c_gpu_h = (float *)malloc(mat_bytes);  // GPU result copied back
    float *c_cpu = (float *)malloc(mat_bytes);     // CPU result

    if (!a_h || !b_h || !c_gpu_h || !c_cpu) {
        fprintf(stderr, "FATAL: host malloc failed for N=%d\n", N);
        return 1;
    }

    srand(42);  // fixed seed for reproducibility
    for (int i = 0; i < N * N; i++) {
        a_h[i] = (float)rand() / RAND_MAX;
        b_h[i] = (float)rand() / RAND_MAX;
    }
    memset(c_gpu_h, 0, mat_bytes);
    memset(c_cpu, 0, mat_bytes);

    // Device memory
    float *a_d, *b_d, *c_d;
    cudaMalloc((void **)&a_d, mat_bytes);
    cudaMalloc((void **)&b_d, mat_bytes);
    cudaMalloc((void **)&c_d, mat_bytes);

    cudaMemcpy(a_d, a_h, mat_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, mat_bytes, cudaMemcpyHostToDevice);
    cudaMemset(c_d, 0, mat_bytes);

    // --- GPU benchmark (CUDA events measure only the kernel, not memcpy) ---
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++)
        dot_2d_gpu(a_d, N, N, N, b_d, N, N, N, c_d, N, N, N, false, false);
    cudaDeviceSynchronize();

    // Benchmark
    double gpu_times[BENCH_ITERS];
    for (int i = 0; i < BENCH_ITERS; i++) {
        cudaMemset(c_d, 0, mat_bytes);
        cudaEventRecord(ev_start, 0);
        dot_2d_gpu(a_d, N, N, N, b_d, N, N, N, c_d, N, N, N, false, false);
        cudaEventRecord(ev_stop, 0);
        cudaEventSynchronize(ev_stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        gpu_times[i] = ms / 1000.0;
    }

    // Copy final GPU result back for validation
    cudaMemcpy(c_gpu_h, c_d, mat_bytes, cudaMemcpyDeviceToHost);

    // --- CPU benchmark ---
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++)
        dot_2d_cpu(a_h, N, N, N, b_h, N, N, N, c_cpu, N, N, N, false, false);

    struct timespec ts_start, ts_end;
    double cpu_times[BENCH_ITERS];
    for (int i = 0; i < BENCH_ITERS; i++) {
        memset(c_cpu, 0, mat_bytes);
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
        dot_2d_cpu(a_h, N, N, N, b_h, N, N, N, c_cpu, N, N, N, false, false);
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        cpu_times[i] = get_elapsed_sec(&ts_start, &ts_end);
    }

    // --- Validation ---
    int K = N;
    float machine_eps = 1.192e-07f;
    float dynamic_rtol = sqrtf((float)K) * machine_eps * 10.0f;
    float dynamic_atol = 1e-5f + (machine_eps * K);
    float max_diff = 0.0f;
    float max_rel_error = 0.0f;
    bool match = true;

    for (int i = 0; i < N * N; i++) {
        float diff = fabsf(c_gpu_h[i] - c_cpu[i]);
        float rel = diff / (fabsf(c_cpu[i]) + 1e-8f);
        if (diff > max_diff) max_diff = diff;
        if (rel > max_rel_error) max_rel_error = rel;
        if (diff > (dynamic_atol + dynamic_rtol * fabsf(c_cpu[i]))) {
            match = false;
            break;
        }
    }

    // --- Compute stats ---
    double gpu_min = gpu_times[0], gpu_sum = 0;
    double cpu_min = cpu_times[0], cpu_sum = 0;
    for (int i = 0; i < BENCH_ITERS; i++) {
        if (gpu_times[i] < gpu_min) gpu_min = gpu_times[i];
        if (cpu_times[i] < cpu_min) cpu_min = cpu_times[i];
        gpu_sum += gpu_times[i];
        cpu_sum += cpu_times[i];
    }
    double gpu_avg = gpu_sum / BENCH_ITERS;
    double cpu_avg = cpu_sum / BENCH_ITERS;

    // TFLOPS: 2*N^3 / time / 1e12
    double flops = 2.0 * (double)N * (double)N * (double)N;
    double gpu_tflops = (flops / gpu_avg) / 1e12;
    double cpu_tflops = (flops / cpu_avg) / 1e12;

    if (json_output) {
        printf("{\n");
        printf("  \"matrix_size\": %d,\n", N);
        printf("  \"bench_iters\": %d,\n", BENCH_ITERS);
        printf("  \"warmup_iters\": %d,\n", WARMUP_ITERS);
        printf("  \"gpu_avg_sec\": %.6f,\n", gpu_avg);
        printf("  \"gpu_min_sec\": %.6f,\n", gpu_min);
        printf("  \"gpu_tflops\": %.2f,\n", gpu_tflops);
        printf("  \"cpu_avg_sec\": %.6f,\n", cpu_avg);
        printf("  \"cpu_min_sec\": %.6f,\n", cpu_min);
        printf("  \"cpu_tflops\": %.2f,\n", cpu_tflops);
        printf("  \"speedup\": %.2f,\n", cpu_avg / gpu_avg);
        printf("  \"validation_passed\": %s,\n", match ? "true" : "false");
        printf("  \"max_abs_diff\": %.8f,\n", max_diff);
        printf("  \"max_rel_error\": %.8f\n", max_rel_error);
        printf("}\n");
    } else {
        printf("=== dot_2d benchmark N=%d ===\n", N);
        printf("GPU: avg=%.6fs  min=%.6fs  TFLOPS=%.2f\n", gpu_avg, gpu_min, gpu_tflops);
        printf("CPU: avg=%.6fs  min=%.6fs  TFLOPS=%.2f\n", cpu_avg, cpu_min, cpu_tflops);
        printf("Speedup (CPU/GPU): %.2fx\n", cpu_avg / gpu_avg);
        printf("Validation: %s (max_diff=%.8f, max_rel=%.8f)\n",
               match ? "PASS" : "FAIL", max_diff, max_rel_error);
    }

    // Cleanup
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    cublasDestroy(get_cublas_handle());
    free(a_h);
    free(b_h);
    free(c_gpu_h);
    free(c_cpu);

    return match ? 0 : 1;
}
