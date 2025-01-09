#include <iostream>
#include <stdexcept> // For std::invalid_argument
#include <iomanip>    // For formatted output
#include <climits>    // For INT_MAX
#include <vector>     // For std::vector
#include "utils.h"    // For printMatrix
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
    vector<vector<int>> result = CPU_floydWarshall(graph, n);

    // Print the result outside of the function
    printMatrix(result, n);

    return 0;
}
