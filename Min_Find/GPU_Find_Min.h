#ifndef GPU_MIN_FINDER_H
#define GPU_MIN_FINDER_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel declaration for finding the minimum using CUDA.
__global__ void FindMinKernel(const int* const current, int* const next, const int n);

// Utility function prototypes.
void initializeArray(int* const arr, const int n);
void printArray(const int* arr, int n);

// Primary GPU function to find the minimum element in the array.
int findminGPU(int* original_array, int n, bool print_arrays);

#endif  // GPU_MIN_FINDER_H
