# CUDA C Project 2 - Optimized Spatial Distance Histogram (SDH) Computing

## Project Description

This project implements an optimized CUDA-based solution for calculating the Spatial Distance Histogram (SDH) of 3D points. The optimization techniques aim to improve performance through parallel computing, shared memory, and other advanced CUDA strategies.

### Key Objectives:
- Efficiently compute SDH for large datasets using GPU acceleration.
- Leverage CUDA techniques like shared memory, coalesced memory access, and atomic operations.
- Measure and analyze the kernel's performance.

## Requirements

### Prerequisites:
- **CUDA Toolkit**: Installed and configured for compiling and running CUDA programs.
- **GPU**: A CUDA-capable GPU for testing and execution.
- **C Compiler**: GCC or a similar compiler for preprocessing.

### Input:
- Number of samples: The total number of 3D points.
- Bucket width: The histogram bucket size.
- Block size: Number of threads per CUDA block.

### Output:
- Calculated SDH as an array.
- Kernel execution time in seconds.

## Implementation Details

### Overview:

The project builds upon the foundational SDH algorithm, implementing significant optimizations:

1. **Kernel Functions**:
   - Optimized to process distances between 3D points in parallel.
   - Utilized grid and block structures for effective task distribution.

2. **Shared Memory Usage**:
   - Minimized memory latency by storing intermediate computations in shared memory.

3. **Memory Access Optimization**:
   - Ensured coalesced memory access patterns to maximize throughput.

4. **Atomic Operations**:
   - Used atomic operations to safely update shared and global memory.

### Compilation:
To compile the program, use the following command:
```bash
nvcc proj2-tkalandarov.cu -o proj2
```

### Execution:
Run the compiled program as follows:
```bash
./proj2 <number_of_samples> <bucket_width> <block_size>
```
Replace the placeholders with appropriate values.

## Example

### Sample Execution:
```bash
./proj2 100000 0.01 256
```
Expected output:
```
SDH Array: [ ... ]
******** Total Running Time of Kernel = 2.0043 sec *******
```

## Performance Analysis

### Metrics Evaluated:
- Execution time for kernel computations.
- Speedup compared to Project 1.
- Memory bandwidth utilization.

### Tools Used:
- CUDA Profiler for detailed performance evaluation.
- Timing functions (e.g., `cudaEventRecord`) to measure kernel execution time.

