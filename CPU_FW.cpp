#include <iostream>
#include <stdexcept> // For std::invalid_argument
#include <iomanip>    // For formatted output
#include <climits>    // For INT_MAX
#include <vector>     // For std::vector

void floydWarshall(std::vector<std::vector<int>>& g, int n)
{
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

    // Output the result
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

int main() {
    int n = 4;
    // Example graph
    std::vector<std::vector<int>> graph = {
        {0, 3, INT_MAX, INT_MAX},
        {2, 0, INT_MAX, 1},
        {INT_MAX, 7, 0, 2},
        {INT_MAX, INT_MAX, 4, 0}
    };

    floydWarshall(graph, n);

    return 0;
}
