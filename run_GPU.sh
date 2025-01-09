#!/bin/bash

# Compile the CUDA code
nvcc -o gpu_fw_out GPU_FW.cu

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Running the program..."
    ./gpu_fw_out
else
    echo "Compilation failed."
fi
