#include <iostream>
#include <stdexcept> // For std::invalid_argument
#include <iomanip>    // For formatted output
#include <climits>    // For INT_MAX
#include <vector>     // For std::vector

using std::vector;

void flattenMatrix(const vector<vector<int>>& g, vector<int>& flattened, int n) {
    flattened.resize(n * n);  // Resize the 1D vector to hold n*n elements

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            flattened[i * n + j] = g[i][j];  // Calculate the index directly
        }
    }
}
void deflattenMatrix(const vector<vector<int>>& g, vector<int>& flattened, int n) {
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