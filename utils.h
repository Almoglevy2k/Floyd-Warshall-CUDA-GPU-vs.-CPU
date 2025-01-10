#pragma once
#include <vector>  // For std::vector

// Function declarations
void flattenMatrix(const std::vector<std::vector<int>>& g, std::vector<int>& flattened, int n);
void deflattenMatrix(std::vector<std::vector<int>>& g, std::vector<int>& flattened, int n);
void printMatrix(const std::vector<std::vector<int>>& dist, int n);
std::vector<std::vector<int>> generateRandomGraph(int n, int maxWeight = 10);
