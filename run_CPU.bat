@echo off
docker run --rm -v C:\Users\almog\floyd-warshall-cuda:/workspace -w /workspace gcc:latest bash -c "chmod +x run_CPU.sh && ./run_CPU.sh"
