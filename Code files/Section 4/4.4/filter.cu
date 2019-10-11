// Filter the contents of an array.
// Uses a scan, followed by a separate kernel to fill the output.
// Example for video 4.4.

#include <assert.h>
#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>

// Standard CUDA API functions
#include <cuda_runtime_api.h>

// CUDA cooperative groups API
#include <cooperative_groups.h>

#include "../utils.h"

__host__ __device__ bool divisible_by_three(int value)
{
  return (value % 3) == 0;
}

const int BLOCK_SIZE = 128;

// Scan using shared memory, within a single block.
__device__ int block_scan(int idata, int shared_data[],
                          cooperative_groups::thread_block block)
{
  // Index into shared memory
  int si = threadIdx.x;
  shared_data[si] = 0;
  si += blockDim.x;
  shared_data[si] = idata;

  for (int offset = 1; offset < blockDim.x; offset *= 2) {
    cooperative_groups::sync(block);
    int t = shared_data[si] + shared_data[si - offset];
    cooperative_groups::sync(block);
    shared_data[si] = t;
  }

  return shared_data[si];
}

// First step of scan: process each block separately
__global__ void scan1(const int *source, int *dest)
{
  // Shared memory buffer. By allocating extra elements we avoid bounds
  // checks on shared memory access.
  __shared__ int shared_data[2 * BLOCK_SIZE];

  // Index into global memory
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data from global memory
  int idata = source[index];

  // Shared memory scan within this block
  int result =
      block_scan(idata, shared_data, cooperative_groups::this_thread_block());

  // Write back to global memory
  dest[index] = result;
}

// Second step of scan: compute prefix sums for each block
__global__ void scan2(const int *dest, int *block_sums, unsigned int count)
{
  // Shared memory buffer. By allocating extra elements we avoid bounds
  // checks on shared memory access.
  __shared__ int shared_data[2 * BLOCK_SIZE];

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  int idata = (index == 0) ? 0 : dest[index * blockDim.x - 1];
  block_sums[index] =
      block_scan(idata, shared_data, cooperative_groups::this_thread_block());
}

// Final step of scan: add block sums to every result.
__global__ void finish_scan(const int *block_sums, int *dest)
{
  __shared__ int block_sum;

  if (threadIdx.x == 0) {
    block_sum = block_sums[blockIdx.x];
  }
  cooperative_groups::sync(cooperative_groups::this_thread_block());

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  dest[index] += block_sum;
}

// Compute prefix sum of source
void scan(const int *source, int *dest, unsigned int count)
{
  int n_blocks1 = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Temporary buffer for kernel
  int *block_sums;
  cudaCheckError(cudaMalloc(&block_sums, n_blocks1 * sizeof(int)));

  // Run the kernel
  scan1<<<n_blocks1, BLOCK_SIZE>>>(source, dest);

  int n_blocks2 = (n_blocks1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
  // If we had multiple blocks here, we'd need a third level of scans to
  // get the final result.
  assert(n_blocks2 == 1);
  scan2<<<n_blocks2, BLOCK_SIZE>>>(dest, block_sums, n_blocks1);

  finish_scan<<<n_blocks1, BLOCK_SIZE>>>(block_sums, dest);

  cudaCheckError(cudaFree(block_sums));
}

// Test predicate for all elements of source. Fill result with a 1 for values
// that satisfy the predicate, and a 0 otherwise.
__global__ void evaluate_predicate(const int *source, int *result,
                                   unsigned int count)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < count) {
    result[index] = divisible_by_three(source[index]) ? 1 : 0;
  }
}

// Copy values that satisfy the predicate from source to result, using the
// indices array to place them in the correct position.
__global__ void fill_output(const int *source, const int *indices, int *result,
                            unsigned int count)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= count) {
    return;
  }

  int value = source[index];
  if (divisible_by_three(value)) {
    // Subtract 1 from index because scan is inclusive (it counts the current
    // element), so the indices array will contain 1-based indices.
    int output_index = indices[index] - 1;
    result[output_index] = value;
  }
}

int main(int argc, char **argv)
{
  // Maximum possible size with two-level scan.
  const unsigned int COUNT = BLOCK_SIZE * BLOCK_SIZE;
  std::unique_ptr<int[]> source(new int[COUNT]);
  std::unique_ptr<int[]> dest(new int[COUNT]);

  // Fill source matrix with some arbitrary test values
  std::mt19937 rng;
  rng.seed(0);
  std::uniform_int_distribution<std::mt19937::result_type> dist(0, 9);

  for (int i = 0; i < COUNT; i++) {
    source[i] = dist(rng);
  }

  // Allocate and fill device memory
  int *source_dev, *dest_dev;
  // Result of evaluating predicates
  int *predicates;
  // Indices at which to store each result element
  int *indices;
  size_t size = COUNT * sizeof(int);
  // Number of elements in the output array
  int output_count;
  cudaCheckError(cudaMalloc(&source_dev, size));
  cudaCheckError(
      cudaMemcpy(source_dev, source.get(), size, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMalloc(&predicates, size));
  cudaCheckError(cudaMalloc(&indices, size));

  {
    KernelTimer t;

    int n_blocks = (COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Test predicate for all source values
    evaluate_predicate<<<n_blocks, BLOCK_SIZE>>>(source_dev, predicates, COUNT);
    // Scan the predicate array to compute output indices
    scan(predicates, indices, COUNT);

    // Find the length of the output from the last index, and allocate the
    // array.
    cudaCheckError(cudaMemcpy(&output_count, indices + COUNT - 1, sizeof(int),
                              cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMalloc(&dest_dev, output_count * sizeof(int)));

    // Copy elements from input to output
    fill_output<<<n_blocks, BLOCK_SIZE>>>(source_dev, indices, dest_dev, COUNT);
  }

  // Copy result back to the host
  cudaCheckError(cudaMemcpy(dest.get(), dest_dev, output_count * sizeof(int),
                            cudaMemcpyDeviceToHost));
  cudaCheckError(cudaFree(source_dev));
  cudaCheckError(cudaFree(dest_dev));
  cudaCheckError(cudaFree(predicates));
  cudaCheckError(cudaFree(indices));

  // Compare with reference implementation
  std::vector<int> dest_reference;
  std::copy_if(source.get(), source.get() + COUNT,
               std::back_inserter(dest_reference), divisible_by_three);
  assert(dest_reference.size() == output_count);
  for (int i = 0; i < output_count; i++) {
    assert(dest_reference[i] == dest.get()[i]);
  }

  return 0;
}
