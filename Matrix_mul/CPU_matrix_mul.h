// CPUmatMul.h

#ifndef CPUMATMUL_H  // If CPUMATMUL_H is not defined
#define CPUMATMUL_H  // Define CPUMATMUL_H

#include <iostream>

// Function to perform matrix multiplication for square matrices
void CPUmatMul(const float *const A, const float *const B, float *const C, const int N);

// Function to test matrix multiplication
float *CPU_mat_mul_test(float *A, float *B, float *R, int N);

#endif  // CPUMATMUL_H    // End of include guard
