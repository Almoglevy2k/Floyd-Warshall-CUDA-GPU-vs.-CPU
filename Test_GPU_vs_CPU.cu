#include "GPU_FW.cu" // Header for GPU implementation
#include "CPU_FW.cpp" // Header for CPU implementation
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

vector<vector<int>> initializeGraph(int n) {
    vector<vector<int>> graph(n, vector<int>(n, INT_MAX));
    for (int i = 0; i < n; i++) {
        graph[i][i] = 0; // Distance to self is 0
        for (int j = 0; j < n; j++) {
            if (i != j) {
                graph[i][j] = rand() % 100 + 1; // Random weights for edges
            }
        }
    }
    return graph;
}

vector<vector<int>> test_GPU(vector<vector<int>> graph, int n) {
    cout << "Input Matrix (GPU):" << endl;
    printMatrix(graph, n);

    // CUDA timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vector<vector<int>> result = GPU_floydWarshall(graph, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "GPU Execution Time: " << milliseconds << " ms" << endl;

    printMatrix(result, n);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

vector<vector<int>> test_CPU(vector<vector<int>> graph, int n) {
    cout << "Input Matrix (CPU):" << endl;
    printMatrix(graph, n);

    auto start = chrono::high_resolution_clock::now();
    vector<vector<int>> result = CPU_floydWarshall(graph, n);
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "CPU Execution Time: " << duration.count() << " ms" << endl;

    printMatrix(result, n);
    return result;
}

int main() {
    int n = 5; // Size of the graph
    vector<vector<int>> graph = initializeGraph(n);

    // Test GPU and CPU implementations
    vector<vector<int>> gpuResult = test_GPU(graph, n);
    vector<vector<int>> cpuResult = test_CPU(graph, n);

    // Validate results
    if (gpuResult == cpuResult) {
        cout << "GPU and CPU results match!" << endl;
    } else {
        cout << "GPU and CPU results do NOT match!" << endl;
    }

    return 0;
}
