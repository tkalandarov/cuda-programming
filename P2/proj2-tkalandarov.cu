/* ==================================================================
  Programmer: Timur Kalandarov (tkalandarov@usf.edu)
  An optimized version of SDH algorithm implementation for 3D data
  To run: `/apps/GPU_course/runScript.sh {name.cu} {#of_samples} {bucket_width} {block_size}`
   ==================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <ctype.h>

#define BOX_SIZE 23000 // size of the data box on one dimension

// definition of single atom
typedef struct atomdesc
{
  double x_pos;
  double y_pos;
  double z_pos;
} atom;

// definition of a bucket
typedef struct hist_entry
{
  unsigned long long distance_count; // need a long long type as the count might be huge
} bucket;

bucket *histogram;                 // list of all buckets in the histogram
unsigned long long PDH_atom_count; // total number of data points
int num_buckets;                   // total number of buckets in the histogram
double PDH_bucket_width;           // value of w
atom *atom_list;                   // list of all data points
int PDH_block_size;                // block size

// These are for an old way of tracking time
struct timezone Idunno;
struct timeval startTime, endTime;

// To track GPU running time
cudaEvent_t start, stop;

// set a checkpoint and show the (natural) running time in seconds
void report_running_time1()
{
  long sec_diff, usec_diff;
  gettimeofday(&endTime, &Idunno);
  sec_diff = endTime.tv_sec - startTime.tv_sec;
  usec_diff = endTime.tv_usec - startTime.tv_usec;
  if (usec_diff < 0)
  {
    sec_diff--;
    usec_diff += 1000000;
  }

  long total_usec = sec_diff * 1000000 + usec_diff;
  double total_ms = total_usec / 1000.0; // Convert to milliseconds

  printf("Running time for CPU version: %0.5f ms\n", total_ms);
}

void report_running_time2()
{
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  printf("Running time for GPU version: %0.5f ms\n", elapsedTime);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

// CUDA Error Check
void checkCudaError(cudaError_t e, const char *in)
{
  if (e != cudaSuccess)
  {
    printf("CUDA Error: %s, %s \n", in, cudaGetErrorString(e));
    exit(EXIT_FAILURE);
  }
}

// distance of two points in the atom_list
double p2p_distance1(atom *a, int index1, int index2)
{
  double x1 = a[index1].x_pos;
  double x2 = a[index2].x_pos;

  double y1 = a[index1].y_pos;
  double y2 = a[index2].y_pos;

  double z1 = a[index1].z_pos;
  double z2 = a[index2].z_pos;

  return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
}

// distance of two points in the atom_list
__device__ double p2p_distance2(atom *a, int index1, int index2)
{
  double x1 = a[index1].x_pos;
  double x2 = a[index2].x_pos;

  double y1 = a[index1].y_pos;
  double y2 = a[index2].y_pos;

  double z1 = a[index1].z_pos;
  double z2 = a[index2].z_pos;

  return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
}

// brute-force SDH solution in a single CPU thread
int PDH_baseline1()
{
  int i, j, h_pos;
  double dist;

  for (i = 0; i < PDH_atom_count; i++)
  {
    for (j = i + 1; j < PDH_atom_count; j++)
    {
      dist = p2p_distance1(atom_list, i, j);
      h_pos = (int)(dist / PDH_bucket_width);
      histogram[h_pos].distance_count++;
    }
  }
  return 0;
}

__global__ void PDH_baseline2(bucket *histogram, atom *atomList, double bucket_width, unsigned long long PDH_atom_count, int num_buckets)
{
  int tidx = threadIdx.x;                        // Get the thread index within the block
  int threadId = blockIdx.x * blockDim.x + tidx; // Calculate the global thread ID

  extern __shared__ int shared_histogram[]; // Declare shared memory for histogram

  // Initialize shared_histogram to 0, each thread handles a different bucket
  for (int i = tidx; i < num_buckets; i += blockDim.x)
  {
    shared_histogram[i] = 0;
  }

  __syncthreads(); // Ensure all threads have initialized shared_histogram

  int i;
  for (i = threadId + 1; i < PDH_atom_count; i++)
  {
    // Calculate the distance between atomList[i] and atomList[threadId]
    double dist = p2p_distance2(atomList, i, threadId);

    // Determine the histogram bucket (bin) for this distance
    int h_pos = (int)(dist / bucket_width);

    // Atomically increment the corresponding bucket in shared_histogram
    atomicAdd(&(shared_histogram[h_pos]), 1);
  }

  __syncthreads(); // Synchronize threads after histogram computation

  // Accumulate the results from shared_histogram into the global histogram
  for (i = tidx; i < num_buckets; i += blockDim.x)
  {
    // Atomically add the value from shared_histogram to the global histogram
    atomicAdd(&(histogram[i].distance_count), shared_histogram[i]);
  }
}

// print the counts in all buckets of the histogram
void output_histogram(bucket *histogram)
{
  int i;
  unsigned long long total_count = 0;
  for (i = 0; i < num_buckets; i++)
  {
    if (i % 5 == 0) // print 5 buckets in a row
      printf("\n%02d: ", i);
    printf("%15lld ", histogram[i].distance_count);
    total_count += histogram[i].distance_count;

    // we also want to make sure the total distance count is correct
    if (i == num_buckets - 1)
      printf("\n T:%lld \n", total_count);
    else
      printf("| ");
  }
}

void output_histogram_diff(bucket *histogram1, bucket *histogram2)
{
  int i;
  unsigned long long total_count = 0;
  unsigned long long diff;
  for (i = 0; i < num_buckets; i++)
  {
    if (i % 5 == 0) /* we print 5 buckets in a row */
      printf("\n%02d: ", i);
    diff = histogram1[i].distance_count - histogram2[i].distance_count;
    printf("%15lld ", diff);
    total_count += histogram1[i].distance_count;

    // we also want to make sure the total distance count is correct
    if (i == num_buckets - 1)
      printf("\n T:%lld \n", total_count);
    else
      printf("| ");
  }
}

// Input Validation
bool isNumber(char number[], bool isFloat)
{
  for (int i = 0; number[i] != 0; i++)
  {
    if (!isdigit(number[i]))
    {
      if ((number[i] == '.' && isFloat))
      {
        isFloat = false;
      }
      else
      {
        return false;
      }
    }
  }
  return true;
}

int main(int argc, char **argv)
{
  int i;

  if (argc != 4)
  {
    printf("Incorrect number of arguments. Expected: %s <Atom Count> <Bucket Width> <Block Size>\n", argv[0]);
    return EXIT_FAILURE;
  }

  if (isNumber(argv[1], false) && isNumber(argv[2], true) && isNumber(argv[3], false))
  {
    PDH_atom_count = atoi(argv[1]);   // number of atoms
    PDH_bucket_width = atof(argv[2]); // bucket width
    PDH_block_size = atof(argv[3]);   // block size
  }
  else
  {
    printf("Incorrect input. Program will close.\n");
    return EXIT_FAILURE;
  }

  bucket *histogram2;

  num_buckets = (int)(BOX_SIZE * 1.732 / PDH_bucket_width) + 1; // number of buckets needed for SDH

  size_t histogramSize = sizeof(bucket) * num_buckets;
  size_t atomSize = sizeof(atom) * PDH_atom_count;

  histogram = (bucket *)malloc(sizeof(bucket) * num_buckets);
  histogram2 = (bucket *)malloc(sizeof(bucket) * num_buckets);
  atom_list = (atom *)malloc(sizeof(atom) * PDH_atom_count);

  srand(1);
  // generate data following a uniform distribution
  for (i = 0; i < PDH_atom_count; i++)
  {
    atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
    atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
    atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
  }

  // Start CPU timer
  gettimeofday(&startTime, &Idunno);

  PDH_baseline1();

  output_histogram(histogram);
  report_running_time1();

  // Malloc space on device, copy to device
  bucket *d_histogram = NULL;
  atom *d_atom_list = NULL;

  checkCudaError(cudaMalloc((void **)&d_histogram, histogramSize),
                 "Malloc Histogram");
  checkCudaError(cudaMalloc((void **)&d_atom_list, atomSize),
                 "Malloc Atom List");

  checkCudaError(cudaMemcpy(d_histogram, histogram2, histogramSize, cudaMemcpyHostToDevice),
                 "Copy histogram to Device");
  checkCudaError(cudaMemcpy(d_atom_list, atom_list, atomSize, cudaMemcpyHostToDevice),
                 "Copy atom_list to Device");

  // Start GPU timer
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // CUDA Kernel Call
  int numBlocks = (PDH_atom_count + PDH_block_size - 1) / PDH_block_size;
  int sharedMemorySize = sizeof(int) * num_buckets;
  PDH_baseline2<<<numBlocks, PDH_block_size, sharedMemorySize>>>(d_histogram, d_atom_list, PDH_bucket_width, PDH_atom_count, num_buckets);

  checkCudaError(cudaGetLastError(), "Kernel Launch");

  checkCudaError(cudaMemcpy(histogram2, d_histogram, histogramSize, cudaMemcpyDeviceToHost),
                 "Copy device histogram to host");

  output_histogram(histogram2);
  report_running_time2();

  // Show differences in two histograms
  output_histogram_diff(histogram, histogram2);

  checkCudaError(cudaFree(d_histogram), "Free device histogram");
  checkCudaError(cudaFree(d_atom_list), "Free device atom_list");

  checkCudaError(cudaDeviceReset(), "Device reset");

  free(histogram);
  free(histogram2);
  free(atom_list);

  return 0;
}