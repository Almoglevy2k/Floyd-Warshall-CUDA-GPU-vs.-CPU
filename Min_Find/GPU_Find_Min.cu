#include "GPU_Find_Min.h"
__global__ void FindMinKernel(const int* const current, int* const next, const int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;  // Global thread ID.

    if (tid * 2 + 1 < n) {                                        // Check if there are two elements to compare
        next[tid] = min(current[tid * 2], current[tid * 2 + 1]);  // Store the minimum of the two elements for the next iteration.
    } else if (tid * 2 < n) {
        next[tid] = current[tid * 2];  // if there is only one element left, store it for the next iteration.
    }
}

void initializeArray(int* const arr, const int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;  // Random number between 0 and 999.
    }
}

void printArray(const int* arr, int n) {
    for (int i = 0; i < n; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void print_while_iterating(const int* current_array, int size) {
    // Allocate host memory for copying the device array
    int* host_array = (int*)malloc(size * sizeof(int));

    // Copy data from device to host
    cudaMemcpy(host_array, current_array, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the array using the existing printArray function
    printArray(host_array, size);

    // Free the temporary host memory
    free(host_array);
}

int findminGPU(int* original_array, int n, bool print_arrays) {
    int size = n;
    initializeArray(original_array, n);

    int* current_array;
    int* next_array;

    cudaMalloc(&current_array, n * sizeof(int));
    cudaMalloc(&next_array, n * sizeof(int));

    cudaMemcpy(current_array, original_array, n * sizeof(int), cudaMemcpyHostToDevice);

    // Create and record start event before the calculation loop.
    cudaEvent_t startEvent, endEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&endEvent);
    cudaEventRecord(startEvent, 0);

    // --- Begin Calculation Loop ---
    while (size > 1) {
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        FindMinKernel<<<gridSize, blockSize>>>(current_array, next_array, size);
        cudaDeviceSynchronize();

        if (print_arrays) {
            print_while_iterating(current_array, size);
        }

        int* temp = current_array;
        current_array = next_array;
        next_array = temp;

        size = (size + 1) / 2;
    }
    // --- End Calculation Loop ---

    // Record end event after calculation loop completes.
    cudaEventRecord(endEvent, 0);
    cudaEventSynchronize(endEvent);

    // Calculate elapsed time for the calculation.
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, endEvent);
    printf("GPU calculation time: %.3f ms\n", elapsedTime);

    // Clean up CUDA events.
    cudaEventDestroy(startEvent);
    cudaEventDestroy(endEvent);

    int result;
    cudaMemcpy(&result, current_array, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(current_array);
    cudaFree(next_array);

    return result;
}
