# Minimum Value Comparison between CPU and GPU

This project compares the performance of finding the minimum value in a large array using both CPU and GPU implementations. It measures and prints the time taken by each approach, demonstrating the advantages and overhead of GPU parallelization.

---

## Files

- **CPU_Find_Min.h / CPU_Find_Min.cpp**  
  Implementation of a sequential CPU algorithm to find the minimum value in an array.

- **GPU_Find_Min.h / GPU_Find_Min.cpp**  
  Implementation of a GPU algorithm using CUDA to find the minimum value via parallel reduction.

- **Test_Find_Min.cu**  
  The main test program which:
  - Initializes a large array with random values.
  - Computes the minimum using both the CPU and GPU.
  - Measures and prints computation times, including isolated GPU kernel time and overall GPU time.
  - Compares results for correctness.

---

## Outpot format (Real result)
```bash
==============================================
Starting Minimum Value Comparison between CPU and GPU
==============================================       

>>> Initializing array with random values...
Array initialization complete.

>>> Computing minimum on CPU...
CPU minimum: 0
Time taken for CPU computation: 1.633 seconds 
----------------------------------------------

>>> Computing minimum on GPU...
GPU calculation time: 59.325 ms
Time taken for GPU computation (including overhead): 6.385 seconds
----------------------------------------------

==============================================
Test PASSED: Both CPU and GPU results match.
==============================================
```

## Algorithm Explanation

### CPU Algorithm: Sequential Scan
- **Approach:**
  - The CPU iterates through each element of the array.
  - It keeps track of the smallest value found during the iteration.
- **Complexity:**
  - The time complexity is O(n), where n is the number of elements.
  - Every element is checked sequentially.

### GPU Algorithm: Parallel Reduction
- **Approach:**
  - The GPU uses a parallel reduction technique to find the minimum:
    - **Pairwise Comparison:**  
      Threads compare pairs of elements simultaneously.
    - **Reduction Steps:**  
      In each step, the number of elements is halved as each thread outputs the smaller of two compared values.
    - **Iterations:**  
      These steps repeat until only one value remains – the minimum.
- **Complexity:**
  - The reduction runs in approximately O(log n) parallel steps.
  - This is significantly faster than the sequential CPU for the core computation, given enough GPU cores.
- **Overhead Considerations:**
  - While the GPU computation itself is fast, overall performance includes overhead from:
    - Memory allocation.
    - Data transfers.
    - Kernel launch latency.


## How to Run

### Prerequisites
- NVIDIA GPU with CUDA support.
- CUDA Toolkit installed.
- C++ compiler with C++14 support or later.

### Compilation

Open a terminal and navigate to the project directory. Compile the code using `nvcc`:

```bash
nvcc Test_Find_Min.cu CPU_Find_Min.cpp GPU_Find_Min.cpp -o test_min_comparison
```
Execution
```bash
./test_min_comparison
```
