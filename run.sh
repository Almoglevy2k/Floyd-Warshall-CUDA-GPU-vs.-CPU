#!/bin/bash

# Compile the code
g++ -o matrix_multiply CPU_FW.cpp

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Running the program..."
    ./matrix_multiply
else
    echo "Compilation failed."
fi
