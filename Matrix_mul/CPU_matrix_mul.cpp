#include "GPU_matrix_mul.h"

// Function to perform matrix multiplication for square matrices
void CPUmatMul(const float *const A, const float *const B, float *const C, const int N) {
    for (int i = 0; i < N; i++) {      // Rows of A
        for (int j = 0; j < N; j++) {  // Columns of B
            float sum = 0.0f;
            for (int l = 0; l < N; l++) {  // Columns of A / Rows of B
                sum += A[i * N + l] * B[l * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

float *CPU_mat_mul_test(float *A, float *B, float *R, int n) {
    CPUmatMul(A, B, R, n);
    return R;
}
