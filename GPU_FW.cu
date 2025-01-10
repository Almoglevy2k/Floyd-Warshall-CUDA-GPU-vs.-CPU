#include "GPU_FW.h"  // Include the header file
#include <iostream>
#include <climits>    // For INT_MAX
#include "utils.h"    // For flattenMatrix, deflattenMatrix, and generateRandomGraph

using std::vector;

// CUDA Kernel for Floyd-Warshall
__global__ void GPU_single_floydWarshall(int* dist, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Global row index
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Global column index

    if (i < n && j < n) {  // Ensure bounds are respected
        if (dist[i * n + k] != INT_MAX && dist[k * n + j] != INT_MAX) {
            int through_k = dist[i * n + k] + dist[k * n + j];
            if (through_k < dist[i * n + j]) {
                dist[i * n + j] = through_k;
            }
        }
    }
}

// Function to run Floyd-Warshall on the GPU
std::vector<std::vector<int>> GPU_floydWarshall(std::vector<std::vector<int>>& g, int n) {
    // Flatten the graph for better memory access and performance
    vector<int> flat_dist;  // Create an empty 1D vector
    flattenMatrix(g, flat_dist, n);  // Flatten the 2D graph into 1D

    // Allocate device memory for the flattened distance matrix
    int* d_flat_dist;
    cudaMalloc((void**)&d_flat_dist, n * n * sizeof(int));  // Allocate memory on the device

    // Copy the flattened matrix from host to device
    cudaMemcpy(d_flat_dist, flat_dist.data(), n * n * sizeof(int), cudaMemcpyHostToDevice);

    // Set up grid and block sizes
    dim3 blockSize(16, 16);  // 16x16 threads per block
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);  // Grid size

    // For every k, run the kernel for all i and j
    for (int k = 0; k < n; k++) {
        GPU_single_floydWarshall<<<gridSize, blockSize>>>(d_flat_dist, n, k);
        cudaDeviceSynchronize();  // Ensure all threads finish before proceeding to the next 'k'
    }

    // Copy the result back to the host
    cudaMemcpy(flat_dist.data(), d_flat_dist, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_flat_dist);

    // Convert the flattened matrix back to 2D
    deflattenMatrix(g, flat_dist, n);  // Convert the 1D array back into 2D
    return g;
}
