#include <iostream>
#include <stdexcept> // For std::invalid_argument
#include <iomanip>    // For formatted output
#include <climits>    // For INT_MAX
#include <vector>     // For std::vector
#include "utils.cpp"    // For printMatrix
#include <chrono>  // For timing
using std::vector;

vector<vector<int>> CPU_floydWarshall(vector<vector<int>>& g, int n)
{
    // Initialize the dist vector with the graph's values
    vector<vector<int>> dist = g;

    // Floyd-Warshall Algorithm
    for (int k = 0; k < n; k++) {  // for every node
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {  // for every pair of nodes
                if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX && dist[i][j] > dist[i][k] + dist[k][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
    // Return the result matrix
    return dist;
}

/*int main() {
    int n = 4;  // Change to a larger value for testing, e.g., n = 1000

    // Use predefined or random graph
    vector<vector<int>> graph = generateRandomGraph(n);  // Replace this with a predefined graph if needed

    // Print the input graph
    std::cout << "Input Matrix:" << std::endl;
    printMatrix(graph, n);

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Run Floyd-Warshall algorithm
    vector<vector<int>> result = CPU_floydWarshall(graph, n);

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate and print the duration in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "CPU Execution Time: " << duration.count() << " ms" << std::endl;

    // Print the result matrix
    std::cout << "Result Matrix:" << std::endl;
    printMatrix(result, n);

    return 0;
}*/
