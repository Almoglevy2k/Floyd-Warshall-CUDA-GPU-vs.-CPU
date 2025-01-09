#include <iostream>
#include <stdexcept> // For std::invalid_argument
#include <iomanip>    // For formatted output
#include <climits>    // For INT_MAX
#include <vector>     // For std::vector
#include <cuda_runtime.h>  // Include CUDA runtime functions (for malloc, memcpy, etc.)
using std::vector;

// CUDA Kernel for Floyd-Warshall
__global__ void GPU_single_floydWarshall(int* dist, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Global index for i
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Global index for j

    // Ensure we stay within bounds
    if (dist[i * n + k] != INT_MAX && dist[k * n + j] != INT_MAX) {
    int through_k = dist[i * n + k] + dist[k * n + j];
    if (through_k < dist[i * n + j]) {
        dist[i * n + j] = through_k;
    }
 }
}

// Function to flatten the 2D matrix into a 1D array
vector<int> flattenMatrix(const vector<vector<int>>& g, int n) {
    vector<int> flat_dist(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            flat_dist[i * n + j] = g[i][j];
        }
    }
    return flat_dist;
}

// Function to deflate the 1D array back into a 2D matrix
void deflattenMatrix(vector<vector<int>>& g, const vector<int>& flat_dist, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            g[i][j] = flat_dist[i * n + j];
        }
    }
}

// Function to run Floyd-Warshall on the GPU
vector<vector<int>> GPU_floydWarshall(vector<vector<int>>& g, int n) {
    // Flatten the graph for better memory access and performance
    vector<int> flat_dist = flattenMatrix(g, n);

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
    deflattenMatrix(g, flat_dist, n);  // Assuming deflattenMatrix converts the 1D back to 2D
    return g;
}

// Function to print the matrix
void printMatrix(const vector<vector<int>>& matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (matrix[i][j] == INT_MAX)
                std::cout << "INF ";
            else
                std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    int n = 4;
    // Example graph
    vector<vector<int>> graph = {
        {0, 3, INT_MAX, INT_MAX},
        {2, 0, INT_MAX, 1},
        {INT_MAX, 7, 0, 2},
        {INT_MAX, INT_MAX, 4, 0}
    };

    // Compute the shortest distances
    vector<vector<int>> result = GPU_floydWarshall(graph, n);

    // Print the result outside of the function
    printMatrix(result, n);  // Assuming printMatrix prints the 2D matrix
    return 0;
}
