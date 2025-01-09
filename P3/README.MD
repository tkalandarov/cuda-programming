# CUDA C Project 3 - Parallel Radix Partition

## Project Description

This project implements a CUDA-based solution for the parallel radix partition algorithm. Radix partition is a mechanism often used in parallel hash joins to reorganize input keys into contiguous memory locations based on their radix values. This implementation focuses on correctness and performance, leveraging CUDA's parallel processing capabilities to execute the algorithm efficiently.

### Key Objectives:
- Compute histograms of radix values in parallel.
- Perform prefix scans for histogram data.
- Reorganize data into partitions with contiguous memory allocation.

## Requirements

### Prerequisites:
- **CUDA Toolkit**: Installed and configured for compiling and running CUDA programs.
- **GPU**: A CUDA-capable GPU for testing and execution.
- **C Compiler**: GCC or a similar compiler for preprocessing.

### Input:
- Number of elements in the array.
- Number of partitions (power of 2).

### Output:
- Partition offsets and number of keys in each partition.
- Total running time of all kernels.

## Implementation Details

### Overview:

The project implements the following CUDA kernels:

1. **Histogram Kernel**:
   - Computes radix histograms for input data using atomic operations.
   - Outputs the number of keys per partition.

2. **Prefix Scan Kernel**:
   - Performs an exclusive prefix sum on the histogram.
   - Determines offsets for data reordering.

3. **Reorder Kernel**:
   - Reorganizes input keys into contiguous partitions using prefix sum offsets.
   - Ensures data consistency through atomic operations.

### Compilation:
To compile the program, use the following command:
```bash
nvcc proj3.cu -o proj3
```

### Execution:
Run the compiled program as follows:
```bash
./proj3 <number_of_elements> <number_of_partitions>
```
Replace the placeholders with appropriate values.

## Example

### Sample Execution:
```bash
./proj3 1000000 8
```
Expected output:
```
Partition 0: Offset = 0, Keys = 125000
Partition 1: Offset = 125000, Keys = 125000
...
******** Total Running Time of All Kernels = 2.0043 sec *******
```

## Performance Analysis

### Metrics Evaluated:
- Execution time for each kernel and total runtime.
- Accuracy of partitioned data.
- Scalability with larger datasets.

### Tools Used:
- CUDA Profiler for analyzing kernel performance.
- CUDA events for precise time measurement of kernels.

