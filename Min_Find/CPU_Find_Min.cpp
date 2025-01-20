#include "CPU_Find_Min.h"

int findMinCPU(const int* arr, int n) {
    int minVal = arr[0];
    for (int i = 1; i < n; ++i) {
        if (arr[i] < minVal) {
            minVal = arr[i];
        }
    }
    return minVal;
}
