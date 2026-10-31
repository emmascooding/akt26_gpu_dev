# CUDA Batched GEMM: Custom Kernel vs cuBLAS

##  Project Overview
This project implements and benchmarks a **Batched General Matrix Multiplication (GEMM)** on NVIDIA GPUs. 

The goal is to compare the performance of a **custom CUDA kernel** (using shared memory) against **NVIDIA's highly optimized `cuBLAS` library** (`cublasSgemmStridedBatched`), while verifying the mathematical correctness of the results against a CPU reference implementation.

##  Features
*   **CPU Reference:** A baseline C++ implementation to verify the accuracy of the GPU results.
*   **Custom GPU Kernel:** A tiled CUDA kernel where each thread block processes one matrix from the batch, utilizing shared memory to optimize memory bandwidth.
*   **cuBLAS Baseline:** Uses the official cuBLAS Strided Batched GEMM function for maximum performance.
*   **Error Checking:** Computes the maximum forward error against the CPU reference to ensure mathematical accuracy.
*   **Performance Profiling:** Uses CUDA Events to measure execution time and calculates GFLOP/s.

##  Project Structure
*   `gemm.cu`: The main C++/CUDA source code containing the kernel, CPU reference, and benchmarking logic.
*   `Makefile`: The compilation script using `nvcc`.

## Prerequisites
*   **NVIDIA GPU** (Compute Capability 7.0 or higher recommended, e.g., Volta, Turing, Ampere).
*   **CUDA Toolkit** (version 11.0 or newer).
*   **C++ Compiler** (GCC or Clang).

##  How to Build and Run

### 1. Compile the code
Use the provided Makefile. If you are using an architecture other than Turing (sm_75), you may need to change the `sm` variable in the Makefile.
```bash
make
