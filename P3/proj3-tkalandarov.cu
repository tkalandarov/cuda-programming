#include <cstdlib>
#include <iostream>
#include <assert.h>
#include <stdio.h>

using namespace std;

#define RAND_RANGE(N) ((double)rand() / ((double)RAND_MAX + 1) * (N))

// Data Generator Function
/* This function generates an array of integers and
   shuffles them using the Knuth shuffle algorithm.
*/
void dataGenerator(long long *data, long long count, int first, int step)
{
  assert(data != NULL);

  for (int i = 0; i < count; ++i)
    data[i] = first + i * step;
  srand(time(NULL));
  for (int i = count - 1; i > 0; i--) // knuth shuffle
  {
    int j = RAND_RANGE(i);
    int k_tmp = data[i];
    data[i] = data[j];
    data[j] = k_tmp;
  }
}

// Bit Field Extraction (BFE) Function
/* This function embeds PTX code of CUDA to extract bit field from x.
   "start" is the starting bit position relative to the LSB.
   "nbits" is the bit field length.
   It returns the extracted bit field as an unsigned integer.
*/
__device__ uint bfe(uint x, uint start, uint nbits)
{
  uint bits;
  asm("bfe.u32 %0, %1, %2, %3;"
      : "=r"(bits)
      : "r"(x), "r"(start), "r"(nbits));
  return bits;
}

// Histogram Kernel Function
/* This function calculates the histogram of radix values for the
   input data using atomic operations.
*/
__global__ void histogram(long long *input_data, int *radix_histogram, int partition_size, long long data_size)
{
  // Calculate the global thread ID using block and thread indices
  long long global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  // Check if the thread ID is within the valid range of the input data
  if (global_thread_id < data_size)
  {
    // Determine the number of bits required for representing the partition size
    uint num_bits = ceil(log2((float)partition_size));

    // Extract the radix value (hash value) for the current input element
    uint radix_value = bfe(input_data[global_thread_id], 0, num_bits);

    // Increment the corresponding bin in the histogram using atomic addition
    atomicAdd(&(radix_histogram[radix_value]), 1);
  }
}

// Prefix Scan Kernel Function
/* This function performs an inclusive parallel prefix sum (scan)
   on the histogram array. It uses shared memory and atomic operations for
   parallel reduction and accumulation.
*/
__global__ void prefixScan(int *input_array, int *prefix_sum_array, int array_size)
{
  // Calculate the global thread ID using block and thread indices
  long long global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int thread_id_in_block = threadIdx.x;

  int stride = 1;

  // Check if the thread ID is within the valid range of the input array
  if (global_thread_id < array_size / 2)
  {
    // Initialize shared memory with adjacent elements from the input array
    input_array[2 * thread_id_in_block] = input_array[2 * thread_id_in_block];
    input_array[2 * thread_id_in_block + 1] = input_array[2 * thread_id_in_block + 1];

    // Parallel reduction using a for loop with bitshift
    for (int i = array_size >> 1; i > 0; i >>= 1)
    {
      // Perform parallel reduction only for active threads
      if (thread_id_in_block < i)
      {
        // Calculate indices for the elements involved in the reduction
        int index_1 = stride * (2 * thread_id_in_block + 1) - 1;
        int index_2 = stride * (2 * thread_id_in_block + 2) - 1;

        // Perform atomic addition to update the value in shared memory
        atomicAdd(&(input_array[index_2]), (input_array[index_1])); // moved shared memory
      }

      stride *= 2;
    }
    __syncthreads();

    // Perform an exclusive scan for the last element in the block
    if (thread_id_in_block == 0)
    {
      input_array[array_size - 1] = 0;
    }

    // Parallel reduction using a for loop for the second phase of the scan
    for (int i = 1; i < array_size; i *= 2)
    {
      stride >>= 1;

      // Perform parallel reduction only for active threads
      if (thread_id_in_block < i)
      {
        // Calculate indices for the elements involved in the reduction
        int index_1 = stride * (2 * thread_id_in_block + 1) - 1;
        int index_2 = stride * (2 * thread_id_in_block + 2) - 1;

        // Swap and perform atomic addition to update the value in shared memory
        int temp = input_array[index_1];
        input_array[index_1] = input_array[index_2];
        atomicAdd(&(input_array[index_2]), temp);
      }
    }

    // Move results into the prefix sum array
    prefix_sum_array[2 * thread_id_in_block] = input_array[2 * thread_id_in_block];
    prefix_sum_array[2 * thread_id_in_block + 1] = input_array[2 * thread_id_in_block + 1];
  }
  __syncthreads();
}

// Reorder Kernel Function
/* This function uses the computed prefix sum to reorder the input
   data based on the radix values. It uses atomic operations to place
   elements in their correct positions in the output array.
*/
__global__ void Reorder(long long *input_data, int *prefix_sum, int partition_size, long long input_size, long long *output_data)
{
  // Calculate the global thread ID using block and thread indices
  long long thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  // Check if the thread ID is within the valid range of input data
  if (thread_id < input_size)
  {
    // Determine the number of bits required for representing the partition size
    uint num_bits = ceil(log2((float)partition_size));

    // Extract the radix value (hash value) for the current input element
    uint radix_value = bfe(input_data[thread_id], 0, num_bits);

    // Perform an atomic increment on the corresponding prefix sum, obtaining the new index
    int new_index = atomicAdd(&(prefix_sum[radix_value]), 1);

    // Place the input element into the reordered output array at the computed index
    output_data[new_index] = input_data[thread_id];
  }
}

int main(int argc, char const *argv[])
{
  // Check if the correct number of command line arguments is provided
  if (argc != 3)
  {
    cerr << "Usage: " << argv[0] << " <data_size> <partition_size>" << endl;
    return 1; // Return an error code
  }

  // Parse command line arguments
  long long data_size;
  int partition_size;

  try
  {
    data_size = atoll(argv[1]);
    partition_size = atoi(argv[2]);

    // Check if non-negative values are provided
    if (data_size <= 0 || partition_size <= 0)
    {
      cerr << "Error: Both data_size and partition_size must be positive integers." << endl;
      return 1; // Return an error code
    }
  }
  catch (const invalid_argument &e)
  {
    cerr << "Error: Invalid argument. Please provide valid positive integers for data_size and partition_size." << endl;
    return 1; // Return an error code
  }
  catch (const out_of_range &e)
  {
    cerr << "Error: Out of range. Please provide valid positive integers within the representable range for data_size and partition_size." << endl;
    return 1; // Return an error code
  }

  // Declare and allocate memory for host variables
  long long *host_input;
  int *host_partition = (int *)malloc(sizeof(int) * partition_size);
  int *host_prefix_scan = (int *)malloc(sizeof(int) * partition_size);
  long long *host_output = (long long *)malloc(sizeof(long long) * data_size);

  // Declare and allocate memory for device variables
  long long *device_input;
  int *device_partition;
  int *device_prefix_scan;
  long long *device_output;

  cudaMallocHost((void **)&host_input, sizeof(long long) * data_size);
  cudaMalloc((void **)&device_input, sizeof(long long) * data_size);

  // Generate input data and copy it to the device
  dataGenerator(host_input, data_size, 0, 1);
  cudaMemcpy(device_input, host_input, sizeof(long long) * data_size, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&device_partition, sizeof(int) * partition_size);
  cudaMemcpy(device_partition, host_partition, sizeof(int) * partition_size, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&device_prefix_scan, sizeof(int) * partition_size);

  cudaMalloc((void **)&device_output, sizeof(long long) * data_size);

  // Determine the number of blocks to be used
  int histogram_blocks = ceil(data_size / (float)32);
  int prefix_scan_blocks = ceil(partition_size / (float)32);

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // Launch histogram kernel
  histogram<<<histogram_blocks, 256, sizeof(int) * partition_size>>>(device_input, device_partition, partition_size, data_size);
  cudaDeviceSynchronize();
  cudaMemcpy(host_partition, device_partition, sizeof(int) * partition_size, cudaMemcpyDeviceToHost);

  // Launch prefix scan kernel
  prefixScan<<<prefix_scan_blocks, 256, sizeof(int) * partition_size>>>(device_partition, device_prefix_scan, partition_size);
  cudaDeviceSynchronize();
  cudaMemcpy(host_prefix_scan, device_prefix_scan, sizeof(int) * partition_size, cudaMemcpyDeviceToHost);

  // Launch reorder kernel
  Reorder<<<histogram_blocks, 32>>>(device_input, device_prefix_scan, partition_size, data_size, device_output);
  cudaDeviceSynchronize();
  cudaMemcpy(host_output, device_output, sizeof(long long) * data_size, cudaMemcpyDeviceToHost);

  // Record the end time
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Print out the prefix and partition information
  printf("\n");
  printf("\n");
  for (int i = 0; i < partition_size; i++)
  {
    printf("partition %d: offset %d, number of keys %d\n", i, host_prefix_scan[i], host_partition[i]);
  }

  // Print out the reordered partitions
  int partition_number = 0;
  for (long long i = 0; i < data_size; i++)
  {
    int j = data_size / partition_size;
    if (i % j == 0)
    {
      printf("\nPartition %d:\n", partition_number);
      partition_number++;
    }
    printf("%lld ", host_output[i]);
    j++;
  }
  printf("\n");
  printf("\n");

  // Print the total running time of all kernels
  printf("******** Total Running Time of Kernel =  %0.5f s *******\n", elapsed_time / 1000);

  // Free allocated memory
  cudaFreeHost(host_input);
  cudaFree(device_input);
  free(host_partition);
  cudaFree(device_partition);
  cudaFree(device_prefix_scan);
  free(host_prefix_scan);

  return 0;
}
