#include "utils.h"  // Include the header file
#include <iostream>
#include <stdexcept>  // For std::invalid_argument
#include <iomanip>     // For formatted output
#include <climits>     // For INT_MAX
#include <random>      // For random number generation

using std::vector;

// Function definitions

void flattenMatrix(const vector<vector<int>>& g, vector<int>& flattened, int n) {
    flattened.resize(n * n);  // Resize the 1D vector to hold n*n elements

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            flattened[i * n + j] = g[i][j];  // Calculate the index directly
        }
    }
}

void deflattenMatrix(vector<vector<int>>& g, vector<int>& flattened, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            g[i][j] = flattened[i * n + j];  // Calculate the index directly
        }
    }
}

void printMatrix(const vector<vector<int>>& dist, int n) {
    std::cout << "Shortest distances between every pair of nodes:" << std::endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (dist[i][j] == INT_MAX)
                std::cout << "INF ";
            else
                std::cout << dist[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

vector<vector<int>> generateRandomGraph(int n, int maxWeight) {
    vector<vector<int>> graph(n, vector<int>(n, INT_MAX));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(1, maxWeight);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                graph[i][j] = 0;  // Distance to itself is 0
            } else if (dist(gen) % 2 == 0) {  // Randomly assign some edges
                graph[i][j] = dist(gen);
            }
        }
    }
    return graph;
}
