// Sum the contents of a binary search tree on the device, using dynamic
// parallelism.
// Example for video 6.7.

#include <assert.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

#include <stdio.h>

// Standard CUDA API functions
#include <cuda_runtime_api.h>

#include "bst-sum-kernels.cuh"
#include "../utils.h"

int main(int argc, char **argv)
{
  const unsigned int COUNT = 128;

  // CUDA needs to reserve some device memory to manage synchronization for
  // nested kernels. If we exceed the maximum reserved depth, our kernel will
  // fail. This setting is sufficient for 128 elements. It should be adjusted
  // if COUNT is changed.
  cudaCheckError(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 8));

  // Create device vector with sequential integers
  thrust::device_vector<int> source(COUNT);
  thrust::sequence(source.begin(), source.end());

  // Build the tree
  Tree *root;
  cudaCheckError(cudaMalloc(&root, sizeof(Tree)));

  // Build the tree from a sorted array
  build_tree<<<1, 1>>>(thrust::raw_pointer_cast(&source[0]), source.size(),
                       root);

  // Reduce
  int *result_dev;
  cudaCheckError(cudaMalloc(&result_dev, sizeof(int)));

  sum_tree<<<1, 1>>>(root, result_dev);

  // Check results
  int result;
  cudaCheckError(
      cudaMemcpy(&result, result_dev, sizeof(int), cudaMemcpyDefault));
  int reference =
      thrust::reduce(source.begin(), source.end(), 0, thrust::plus<int>());

  printf("Sum of %u elements: %d\n", COUNT, result);
  assert(result == reference);

  // Clean up
  destroy_tree<<<1, 1>>>(root);
  cudaCheckError(cudaFree(root));

  cudaCheckError(cudaDeviceSynchronize());
}
