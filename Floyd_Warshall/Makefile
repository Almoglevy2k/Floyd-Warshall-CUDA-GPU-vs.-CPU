# Define directories
SRC_DIR := src
BUILD_DIR := build

# Define source files
SRC_FILES := $(SRC_DIR)/GPU_FW.cu $(SRC_DIR)/CPU_FW.cpp $(SRC_DIR)/Test_GPU_vs_CPU.cu $(SRC_DIR)/utils.cpp

# Output binary
OUTPUT := $(BUILD_DIR)/floyd_warshall_test

# Compile command
NVCC := nvcc -ccbin g++

# Default target
all: $(OUTPUT)

$(OUTPUT): $(SRC_FILES)
	$(NVCC) -o $(OUTPUT) $(SRC_FILES) -lcudart

# Clean build
clean:
	rm -rf $(BUILD_DIR)/*

.PHONY: all clean
