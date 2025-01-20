@echo off
docker run --rm -it --gpus all -v /c/Users/Almog/Algorithms_GPU_vs_CPU:/workspace nvidia/cuda:12.2.0-devel-ubuntu20.04 /bin/bash

