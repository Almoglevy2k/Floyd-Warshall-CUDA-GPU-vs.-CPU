# Project Overview

This repository is a collection of implementations and experiments in CUDA and C++, focusing on GPU programming and parallel computation.
The goal of this project is to deepen my understanding of high-performance computing while exploring efficient solutions to computationally intensive problems.

## Testing and Performance Tracking
Each implementation includes a comparison with a CPU-based version of the same algorithm. The tests track metrics such as:
- Execution time for both CPU and GPU implementations.
- Speedup achieved by leveraging GPU parallelism.
- Accuracy and correctness of the results.

## Directory Structure
This project is organized into three main directories, each focusing on a specific algorithm:

- **`Floyd_Warshall/`**: Implements the Floyd-Warshall algorithm for finding the shortest paths in a weighted graph, optimized for GPU execution.
- **`Matrix_mul/`**: Contains a comparison of matrix multiplication using CUDA and standard C++ implementations, demonstrating the power of parallelism.
- **`Min_Find/`**: Features a parallel implementation for finding the minimum value in a large dataset using CUDA.


## How to Navigate
Each directory includes its own `README.md` with detailed explanations, usage instructions, and examples:
- [Floyd_Warshall/README.md](Floyd_Warshall/README.md)
- [Matrix_mul/README.md](Matrix_mul/README.md)
- [Min_Find/README.md](Min_Find/README.md)
