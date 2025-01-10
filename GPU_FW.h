#pragma once
#include <vector>  // For std::vector

// Function declarations
std::vector<std::vector<int>> GPU_floydWarshall(std::vector<std::vector<int>>& g, int n);

// CUDA Kernel function declaration
__global__ void GPU_single_floydWarshall(int* dist, int n, int k);
