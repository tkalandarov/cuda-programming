# CUDA C Project 1 - Parallel System Implementation

## Project Description

This project explores the implementation of a massive parallel system using CUDA C to solve computational problems efficiently by leveraging GPU acceleration. The main objective is to use CUDA to achieve high performance by parallelizing tasks typically constrained by CPU limitations.

### Key Objectives:
- Understand and implement CUDA kernels to optimize problem-solving.
- Utilize shared memory and thread synchronization to maximize performance.
- Analyze computational performance metrics, including execution time and efficiency.

## Requirements

### Prerequisites:
- **CUDA Toolkit**: Installed and configured for compiling and running CUDA C programs.
- **GPU**: A CUDA-capable GPU for testing and execution.
- **C Compiler**: GCC or a similar compiler for preprocessing.

### Input:
- Problem-specific data provided as arrays or matrices, detailed in the project description.

### Output:
- Computed results based on the specific problem (e.g., matrix multiplication, data transformation).
- Performance metrics for the computation.

## Implementation Details

### Overview:
The project implements the following CUDA concepts:

1. **Kernel Functions**:
   - Custom kernel functions designed to process data in parallel.
   - Defined grid and block structures to distribute tasks across GPU threads.

2. **Shared Memory Usage**:
   - Optimized access to memory by utilizing shared memory for intermediate calculations.
   - Reduced global memory access latency.

3. **Thread Synchronization**:
   - Employed `__syncthreads()` to coordinate threads within a block.
   - Ensured correctness of data sharing among threads.

### Code Highlights:
- **Input Parsing**: The program reads data input (details provided in the implementation).
- **Kernel Launch**: CUDA kernels are launched with specified grid and block dimensions to maximize utilization.
- **Post-Processing**: Results from the device are transferred back to the host for analysis.

### Compilation:
To compile the program, use the following command:
```bash
nvcc proj1-tkalandarov.cu -o proj1
```

### Execution:
Run the compiled program as follows:
```bash
./proj1 <input_file>
```
Replace `<input_file>` with the path to the file containing input data.

## Example

### Input Format:
- Provide inputs as described in the project PDF, formatted as matrices or arrays.

### Sample Execution:
```bash
./proj1 input_data.txt
```
Expected output:
```
Result matrix:
...
Execution Time: 0.023s
```

## Performance Analysis

### Metrics Evaluated:
- Execution time for kernel computations.
- Speedup compared to a CPU implementation.
- Memory bandwidth utilization.

### Tools Used:
- CUDA Profiler for detailed performance evaluation.
- Timing functions (e.g., `cudaEventRecord`) to measure kernel execution time.

