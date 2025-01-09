# Variables
TARGET = matrix_multiply
SOURCE = CPU_FW.cpp

# Default target to compile and run
all: build run

# Compile the program
build:
	    g++ -o $(TARGET) $(SOURCE)

# Run the program
run:
	    ./$(TARGET)

# Clean up compiled files
clean:
	    rm -f $(TARGET)
