#include "GPU_matrix_mul.h"

// CUDA Kernel for Matrix Multiplication
__global__ void GPUMatMul(const float* A, const float* B, float* R, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;

    if (row < n && col < n) {
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * n + col];
        }
        R[row * n + col] = sum;
    }
}

// Error-Handling Function Implementation
void handleCudaError(cudaError_t err, const char* context, float* d_A, float* d_B, float* d_R) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error during %s: %s\n", context, cudaGetErrorString(err));
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_R) cudaFree(d_R);
        exit(EXIT_FAILURE);
    }
}

// GPU Matrix Multiplication Test Function
float* GPU_mat_mul_test(const float* A, const float* B, float* R, int n, float* kernel_time_ms) {
    size_t mat_size = sizeof(float) * n * n;

    // Allocate Device Memory
    float *d_A, *d_B, *d_R;
    handleCudaError(cudaMalloc(&d_A, mat_size), "allocating d_A");
    handleCudaError(cudaMalloc(&d_B, mat_size), "allocating d_B");
    handleCudaError(cudaMalloc(&d_R, mat_size), "allocating d_R");

    // Copy Data to Device
    handleCudaError(cudaMemcpy(d_A, A, mat_size, cudaMemcpyHostToDevice), "copying A to d_A", d_A, d_B, d_R);
    handleCudaError(cudaMemcpy(d_B, B, mat_size, cudaMemcpyHostToDevice), "copying B to d_B", d_A, d_B, d_R);

    // Configure Kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    // Start CUDA Event Timing for Kernel Execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    GPUMatMul<<<gridSize, blockSize>>>(d_A, d_B, d_R, n);
    handleCudaError(cudaGetLastError(), "kernel launch", d_A, d_B, d_R);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate Kernel Execution Time
    cudaEventElapsedTime(kernel_time_ms, start, stop);

    // Cleanup CUDA Events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy Results Back to Host
    handleCudaError(cudaMemcpy(R, d_R, mat_size, cudaMemcpyDeviceToHost), "copying d_R to R", d_A, d_B, d_R);

    // Free Device Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_R);

    return R;
}
