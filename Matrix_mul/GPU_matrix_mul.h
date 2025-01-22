#ifndef GPU_MATRIX_MUL_H
#define GPU_MATRIX_MUL_H

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

// CUDA kernel for matrix multiplication
__global__ void GPUMatMul(const float* A, const float* B, float* R, int n);

// Function to handle CUDA errors with optional cleanup
void handleCudaError(cudaError_t err, const char* context, float* d_A = nullptr, float* d_B = nullptr, float* d_R = nullptr);

// GPU Matrix Multiplication Test Function Declaration
float* GPU_mat_mul_test(const float* A, const float* B, float* R, int n, float* kernel_time_ms);

#endif  // GPU_MATRIX_MUL_H
