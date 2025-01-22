#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>

#include "CPU_matrix_mul.h"
#include "GPU_matrix_mul.h"

using namespace std::chrono;

bool compare_matrices(float *A, float *B, int n) {
    for (int i = 0; i < n * n; i++) {
        if (fabs(A[i] - B[i]) > 0.001) {  // floating point comparison
            return false;
        }
    }
    return true;
}

void init_matrix(float *a, int n) {
    for (int i = 0; i < n * n; i++) {
        a[i] = static_cast<float>(rand() % 100) / 100;
    }
}

void simple_init_matrix(float *a, int n) {
    for (int i = 0; i < n * n; i++) {
        a[i] = 1.000;
    }
}

int main() {
    printf("==============================================\n");
    printf("Starting Matrix Multiplication on CPU and GPU\n");
    printf("==============================================\n");

    int n = 2000;  // Matrix size
    float *A, *B, *R_GPU, *R_CPU;

    // Allocate memory for matrices
    try {
        A = new float[n * n];
        B = new float[n * n];
        R_GPU = new float[n * n];
        R_CPU = new float[n * n];
    } catch (const std::exception &e) {
        printf("Memory allocation failed.\n");
        return -1;
    }

    printf("\n>>> Initializing matrices with random values...\n");
    init_matrix(A, n);
    init_matrix(B, n);
    printf("Matrix initialization complete.\n");

    // CPU Timing
    printf("\n>>> Computing matrix multiplication on CPU...\n");
    auto cpu_start = high_resolution_clock::now();
    CPU_mat_mul_test(A, B, R_CPU, n);
    auto cpu_end = high_resolution_clock::now();
    double cpu_time = duration_cast<duration<double>>(cpu_end - cpu_start).count();
    printf("CPU computation complete.\n");
    printf("Time taken for CPU computation: %.3f seconds\n", cpu_time);
    printf("----------------------------------------------\n");

    // GPU Timing
    printf("\n>>> Computing matrix multiplication on GPU...\n");
    float kernel_time_ms = 0.0f;

    auto gpu_overall_start = high_resolution_clock::now();
    GPU_mat_mul_test(A, B, R_GPU, n, &kernel_time_ms);
    auto gpu_overall_end = high_resolution_clock::now();

    double gpu_overall_time = duration_cast<duration<double>>(gpu_overall_end - gpu_overall_start).count();

    printf("GPU computation complete.\n");
    printf("GPU calculation time (kernel only): %.3f ms\n", kernel_time_ms);
    printf("Time taken for GPU computation (including overhead): %.3f seconds\n", gpu_overall_time);
    printf("----------------------------------------------\n");

    // Verify Results
    printf("\n==============================================\n");
    printf("Verifying results between CPU and GPU...\n");

    if (compare_matrices(R_GPU, R_CPU, n)) {
        printf("Test PASSED: Both CPU and GPU results match.\n");
    } else {
        printf("Test FAILED: CPU and GPU results do not match.\n");
    }
    printf("==============================================\n");

    // Clean up
    delete[] A;
    delete[] B;
    delete[] R_GPU;
    delete[] R_CPU;

    return 0;
}
