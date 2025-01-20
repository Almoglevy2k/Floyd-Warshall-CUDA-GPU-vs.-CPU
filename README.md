# Floyd-Warshall CUDA Implementation

This project implements the **Floyd-Warshall algorithm** to find the shortest paths between all pairs of nodes in a graph, with both **CPU** and **GPU** implementations.

## Features

- **CPU Implementation**: Sequential algorithm in C++.
- **GPU Implementation**: Parallelized using CUDA.
- **[Article About the Code](./ Floyd-Warshall_CPU_vs_CPU.pdf)**: A detailed explanation of how the code works and the design decisions behind it.

## Functions

| Function Name         | Description                                      |
|-----------------------|--------------------------------------------------|
| `CPU_floydWarshall`   | CPU-based Floyd-Warshall algorithm.              |
| `GPU_floydWarshall`   | GPU-accelerated Floyd-Warshall using CUDA.       |
| `flattenMatrix`       | Flattens a 2D matrix into a 1D array for GPU.    |
| `deflattenMatrix`     | Converts a 1D array back to a 2D matrix.         |
| `printMatrix`         | Prints the distance matrix.                      |
| `generateRandomGraph` | Generates a random weighted graph.               |

## Requirements

- **CUDA** (for GPU implementation)
- **C++** compiler
- **Docker** (for running the project in a container with GPU support)
- **Make** (for building the project, optional)

## How to Build and Run

### 1. **Using Docker** (Recommended):
   - Ensure **Docker** is installed with GPU support.
   - Run the following command to compile and run the project inside a Docker container:
   
   ```bash
   ./scripts/run_test.sh  # For Linux/WSL
   scripts\run_test.bat  # For Windows
### 2.Using make(optional)
-Install Make and CUDA on your system.
-Run the following commands to build and execute the project: 

```bash
make
./build/floyd_warshall_test

##License
-This project is open-source and available under the MIT License.


Output example - 
```bash
Input Matrix (GPU):
GPU Execution Time: 461.785 ms
Input Matrix (CPU):
CPU Execution Time: 13122 ms
GPU and CPU results match!
```