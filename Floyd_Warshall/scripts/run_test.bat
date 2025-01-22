@echo off
echo Running Test_GPU_vs_CPU.cu inside Docker...

:: Run Docker with GPU access and mount the project directory
docker run --rm --gpus all ^
    -v C:\Users\almog\floyd-warshall-cuda:/workspace ^
    -w /workspace ^
    nvidia/cuda:12.2.0-devel-ubuntu22.04 bash -c "make && ./build/floyd_warshall_test"

:: Check if Docker execution was successful
if %ERRORLEVEL% neq 0 (
    echo Error: Docker execution failed.
    exit /b %ERRORLEVEL%
)

echo Test execution completed successfully.
