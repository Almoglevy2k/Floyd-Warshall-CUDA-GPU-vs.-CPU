# Matrix Multiplication: CPU vs GPU

## Project Overview
This project implements matrix multiplication using both **CPU** and **GPU** to compare their performance. The GPU implementation leverages CUDA for parallel computation, while the CPU uses a standard nested loop approach. Timing analysis is conducted to evaluate the speed difference between the two methods.

---

## Features
- **Matrix Multiplication**:
  - GPU implementation using CUDA.
  - CPU implementation using nested loops.
- **Timing Analysis**:
  - Kernel execution time (GPU).
  - Overall GPU computation time, including memory allocation and data transfer.
- **Result Verification**:
  - Ensures that CPU and GPU results match within a small error margin for floating-point arithmetic.

## Requirements

-**CUDA Toolkit: Ensure you have the CUDA toolkit installed on your system.**

-**C++ Compiler: Required to compile the CPU implementation.**

-**NVIDIA GPU: Ensure your system has an NVIDIA GPU with CUDA support.**
---

## File Structure
| File                 | Description                                        |
|----------------------|----------------------------------------------------|
| `GPU_matrix_mul.cu`  | GPU implementation of matrix multiplication using CUDA. |
| `CPU_matrix_mul.cpp` | CPU implementation of matrix multiplication.       |
| `Test_mat_mul.cu`    | Main testing file for both CPU and GPU implementations. |
| `GPU_matrix_mul.h`   | Header file for GPU-related functions.             |

---

## How to Build and Run
1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
     ```
2. **Compile the Code:Use nvcc to compile the files:**:
```bash
nvcc GPU_matrix_mul.cu Test_mat_mul.cu CPU_matrix_mul.cpp -o matrix_mul
```
3. **Run the Program Using Docker:Ensure you have Docker installed and run the program inside a container to avoid compatibility issues.**:
```bash
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace nvidia/cuda:11.8-base ./matrix_mul
```

## Example Output
```bash
==============================================
Starting Matrix Multiplication on CPU and GPU
==============================================

>>> Initializing matrices with random values...
Matrix initialization complete.

>>> Computing matrix multiplication on CPU...
CPU computation complete.
Time taken for CPU computation: 19.581 seconds
----------------------------------------------

>>> Computing matrix multiplication on GPU...
GPU computation complete.
GPU calculation time (kernel only): 14.109 ms
Time taken for GPU computation (including overhead): 0.181 seconds
----------------------------------------------

==============================================
Verifying results between CPU and GPU...
Test PASSED: Both CPU and GPU results match.
==============================================
```
##Performance Highlights
## Features
- **CPU Computation:**:
  - equential execution using nested loops.
  - Suitable for small matrix sizes but slow for larger matrices.
- **GPU Computation:**:
  - Parallel execution using CUDA.
  - Significantly faster for larger matrices due to massive parallelism.




##  Possible Improvements and Optimizations

- **Shared Memory:**
  - Implement shared memory for temporary storage to minimize global memory accesses.
  - Use tiling techniques to efficiently load matrix sub-blocks into shared memory.

- **Memory Optimization:**
  - Ensure memory coalescing by aligning data accesses for all threads.
  - Reduce redundant reads/writes by caching frequently accessed data in shared memory.

- **Thread Block Optimization:**
  - Experiment with thread block and grid sizes to achieve maximum occupancy of GPU cores.
  - Adjust block size to match the dimensions of the matrix to improve warp utilization.

- **Load Balancing:**
  - Distribute computation evenly across threads to prevent underutilized GPU cores.
  - Avoid thread divergence by ensuring uniform workload among threads in a warp.

- **Performance Profiling:**
  - Use CUDA profiling tools to identify bottlenecks.
  - Optimize kernel launch configuration, memory usage, and compute intensity based on profiling results.



This project is licensed under the MIT License. Feel free to use and modify it as needed.