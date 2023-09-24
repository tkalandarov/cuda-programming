/* ==================================================================
  Programmer: Timur Kalandarov (tkalandarov@usf.edu)
  The basic SDH algorithm implementation for 3D data
  To run: `/apps/GPU_course/runScript.sh /home/t/tkalandarov/COP4520/P1/SDH.cu 10000 500`
   ==================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
__device__ double p2p_distance(atom *a, int index1, int index2)
{
  double x1 = a[index1].x_pos;
  double x2 = a[index2].x_pos;

  double y1 = a[index1].y_pos;
  double y2 = a[index2].y_pos;

  double z1 = a[index1].z_pos;
  double z2 = a[index2].z_pos;

  return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
}

__global__ void PDH_baseline(bucket *histogram, atom *atomList, double bucket_width, unsigned long long PDH_atom_count)
{
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadId >= PDH_atom_count)
    return;

  for (int i = threadId + 1; i < PDH_atom_count; i++)
  {
    double dist = p2p_distance(atomList, threadId, i);
    int pos = (int)(dist / bucket_width);

    atomicAdd(&histogram[pos].distance_count, 1);
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

int main(int argc, char **argv)
{
  int i;

  PDH_atom_count = atoi(argv[1]);   // number of atoms
  PDH_bucket_width = atof(argv[2]); // input distance: bucket width

  num_buckets = (int)(BOX_SIZE * 1.732 / PDH_bucket_width) + 1; // number of buckets needed for SDH

  size_t histogramSize = sizeof(bucket) * num_buckets;
  size_t atomSize = sizeof(atom) * PDH_atom_count;

  histogram = (bucket *)malloc(sizeof(bucket) * num_buckets);
  atom_list = (atom *)malloc(sizeof(atom) * PDH_atom_count);

  srand(1);
  // generate data following a uniform distribution
  for (i = 0; i < PDH_atom_count; i++)
  {
    atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
    atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
    atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
  }

  // Malloc space on device, copy to device
  bucket *d_histogram = NULL;
  atom *d_atom_list = NULL;

  checkCudaError(cudaMalloc((void **)&d_histogram, histogramSize),
                 "Malloc Histogram");
  checkCudaError(cudaMalloc((void **)&d_atom_list, atomSize),
                 "Malloc Atom List");

  checkCudaError(cudaMemcpy(d_histogram, histogram, histogramSize, cudaMemcpyHostToDevice),
                 "Copy histogram to Device");
  checkCudaError(cudaMemcpy(d_atom_list, atom_list, atomSize, cudaMemcpyHostToDevice),
                 "Copy atom_list to Device");

  // CUDA Kernel Call
  PDH_baseline<<<ceil(PDH_atom_count / 32), 32>>>(d_histogram, d_atom_list, PDH_bucket_width, PDH_atom_count);
  checkCudaError(cudaGetLastError(), "Kernel Launch");

  checkCudaError(cudaMemcpy(histogram, d_histogram, histogramSize, cudaMemcpyDeviceToHost),
                 "Copy device histogram to host");

  output_histogram(histogram);

  checkCudaError(cudaFree(d_histogram), "Free device histogram");
  checkCudaError(cudaFree(d_atom_list), "Free device atom_list");

  free(histogram);
  free(atom_list);

  checkCudaError(cudaDeviceReset(), "Device reset");

  return 0;
}
