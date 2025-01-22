#include "CPU_FW.h"  // Include the header file
#include <iostream>
#include <stdexcept>  // For std::invalid_argument
#include <iomanip>     // For formatted output
#include <climits>     // For INT_MAX
#include <chrono>      // For timing
#include "utils.h"     // For printMatrix (header file for utils functions)
using std::vector;

std::vector<std::vector<int>> CPU_floydWarshall(std::vector<std::vector<int>>& g, int n) {
    // Initialize the dist vector with the graph's values
    std::vector<std::vector<int>> dist = g;

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
