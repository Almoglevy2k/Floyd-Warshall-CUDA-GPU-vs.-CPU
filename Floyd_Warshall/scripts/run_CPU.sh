#!/bin/bash

# Compile the code
g++ -o cpu_fw_out CPU_FW.cpp

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Running the program..."
    ./cpu_fw_out
else
    echo "Compilation failed."
fi
