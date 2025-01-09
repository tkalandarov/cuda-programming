# COP 4520 - Computing in Massively Parallel Systems, Fall 2023

This repository contains projects developed for **COP 4520 - Computing in Massively Parallel Systems**, a course focused on the principles and practices of large-scale parallel computing. The course explores CUDA programming basics, advanced parallel code optimization techniques, and domain-specific case studies. Below is an overview of the three projects implemented during the course, highlighting their objectives and differences.

## Projects Overview

### 1. Project 1: Basic CUDA Implementation

**Objective:**
- To gain familiarity with CUDA programming and GPU acceleration by solving a simple computational problem.
- Implement basic CUDA concepts like kernel functions, thread synchronization, and shared memory.

**Key Features:**
- Focused on building foundational skills in GPU programming.
- Explored performance measurement techniques, including CUDA event timing.

**Outcome:**
A basic parallel implementation of a computational task, laying the groundwork for more advanced projects.

---

### 2. Project 2: Optimized Spatial Distance Histogram (SDH) Computation

**Objective:**
- Optimize a computationally intensive problem involving the computation of a spatial distance histogram for 3D points.
- Apply advanced CUDA techniques to improve runtime performance, including memory coalescing, thread workload management, and atomic operations.

**Key Features:**
- Introduced shared memory usage and coalesced memory access patterns.
- Focused on achieving significant speedup over the baseline implementation in Project 1.

**Outcome:**
An optimized solution demonstrating the importance of efficient memory access and workload balancing in GPU programming.

---

### 3. Project 3: Parallel Radix Partition

**Objective:**
- Implement a parallel radix partition algorithm, commonly used in parallel hash joins, using CUDA.
- Reorganize input data into contiguous memory partitions based on radix values.

**Key Features:**
- Utilized histogram computation, prefix scan, and data reordering kernels.
- Focused on correctness and scalability rather than peak performance.
- Explored techniques for managing large datasets and multiple partitions.

**Outcome:**
A robust parallel radix partition implementation with support for scalability and partitioning flexibility.

## How the Projects Differ

1. **Complexity:**
   - Project 1 introduces basic concepts and focuses on understanding CUDA programming fundamentals.
   - Project 2 incorporates advanced optimization techniques to improve performance.
   - Project 3 involves implementing a domain-specific algorithm with a focus on correctness and scalability.

2. **Techniques Used:**
   - Project 1 uses basic CUDA features like kernel launches and thread synchronization.
   - Project 2 emphasizes memory optimization and workload distribution.
   - Project 3 leverages a combination of histogram computation, prefix scan, and data reordering.

3. **Focus:**
   - Project 1: Learning CUDA programming basics.
   - Project 2: Performance optimization and advanced techniques.
   - Project 3: Domain-specific implementation and scalability.

## Repository Structure
- `P1/`: Contains source code, README, and supporting files for Project 1.
- `P2/`: Contains source code, README, and supporting files for Project 2.
- `P3/`: Contains source code, README, and supporting files for Project 3.


