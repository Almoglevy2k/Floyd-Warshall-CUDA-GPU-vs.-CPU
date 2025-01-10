#!/bin/bash

# Ensure GPU support by running inside a Docker container
echo "Compiling and running test inside Docker..."

docker run --rm --gpus all \
    -v "$(pwd)":/workspace \
    -w /workspace \
    nvidia/cuda:12.2.0-devel-ubuntu22.04 bash -c "make && ./build/floyd_warshall_test"

# Check if the command was successful
if [ $? -ne 0 ]; then
    echo "Error: Docker execution failed."
    exit 1
else
    echo "Test completed successfully."
fi
