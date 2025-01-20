#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "CPU_Find_Min.h"
#include "GPU_Find_Min.h"

extern void initializeArray(int* const arr, const int n);

int main() {
    int n = 1000000000;  // Large array size for noticeable performance differences
    int seed = (int)time(NULL);

    // Decorative header
    printf("==============================================\n");
    printf("Starting Minimum Value Comparison between CPU and GPU\n");
    printf("==============================================\n\n");

    // Allocate arrays for CPU and GPU processing.
    int* array_cpu = (int*)malloc(n * sizeof(int));
    int* array_gpu = (int*)malloc(n * sizeof(int));
    if (!array_cpu || !array_gpu) {
        fprintf(stderr, "Memory allocation failed.\n");
        return EXIT_FAILURE;
    }

    // Initialize and randomize the array.
    printf(">>> Initializing array with random values...\n");
    initializeArray(array_cpu, n);
    printf("Array initialization complete.\n\n");

    // Copy the randomized CPU array to the GPU array for consistency.
    for (int i = 0; i < n; ++i) {
        array_gpu[i] = array_cpu[i];
    }

    // Measure CPU computation
    printf(">>> Computing minimum on CPU...\n");
    clock_t start_cpu = clock();
    int cpu_min = findMinCPU(array_cpu, n);
    clock_t end_cpu = clock();
    double cpu_time = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
    printf("CPU minimum: %d\n", cpu_min);
    printf("Time taken for CPU computation: %.3f seconds\n", cpu_time);
    printf("----------------------------------------------\n\n");

    // Measure overall GPU computation time (including overhead).
    printf(">>> Computing minimum on GPU...\n");
    clock_t start_gpu_overall = clock();
    int gpu_min = findminGPU(array_gpu, n, false);
    clock_t end_gpu_overall = clock();
    double gpu_overall_time = (double)(end_gpu_overall - start_gpu_overall) / CLOCKS_PER_SEC;

    // Note: findminGPU internally prints the isolated GPU calculation time.
    printf("Time taken for GPU computation (including overhead): %.3f seconds\n", gpu_overall_time);
    printf("----------------------------------------------\n\n");

    // Compare and print final result
    printf("==============================================\n");
    if (cpu_min == gpu_min) {
        printf("Test PASSED: Both CPU and GPU results match.\n");
    } else {
        printf("Test FAILED: CPU and GPU results do not match.\n");
    }
    printf("==============================================\n");

    // Clean up.
    free(array_cpu);
    free(array_gpu);
    return 0;
}