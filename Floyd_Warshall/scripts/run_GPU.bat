@echo off
docker run --rm --gpus all -v C:\Users\almog\floyd-warshall-cuda:/workspace -w /workspace nvidia/cuda:12.2.0-devel-ubuntu22.04 bash -c "chmod +x run_GPU.sh && ./run_GPU.sh"
