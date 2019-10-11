// Implementation of parallel prefix sum, aka scan.
// Example for video 4.3.

#include <assert.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>

// Standard CUDA API functions
#include <cuda_runtime_api.h>

// CUDA cooperative groups API
#include <cooperative_groups.h>

#include "../utils.h"

void scan_reference(const int *source, int *dest, unsigned int count)
{
  int sum = 0;
  for (int i = 0; i < count; i++) {
    sum += source[i];
    dest[i] = sum;
  }
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
  size_t size = COUNT * sizeof(int);
  cudaCheckError(cudaMalloc(&source_dev, size));
  cudaCheckError(
      cudaMemcpy(source_dev, source.get(), size, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMalloc(&dest_dev, size));

  int n_blocks1 = (COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Temporary buffer for kernel
  int *block_sums;
  cudaCheckError(cudaMalloc(&block_sums, n_blocks1 * sizeof(int)));

  {
    KernelTimer t;

    // Run the kernel
    scan1<<<n_blocks1, BLOCK_SIZE>>>(source_dev, dest_dev);

    int n_blocks2 = (n_blocks1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // If we had multiple blocks here, we'd need a third level of scans to
    // get the final result.
    assert(n_blocks2 == 1);
    scan2<<<n_blocks2, BLOCK_SIZE>>>(dest_dev, block_sums, n_blocks1);

    finish_scan<<<n_blocks1, BLOCK_SIZE>>>(block_sums, dest_dev);
  }

  // Copy result back to the host
  cudaCheckError(
      cudaMemcpy(dest.get(), dest_dev, size, cudaMemcpyDeviceToHost));
  cudaCheckError(cudaFree(source_dev));
  cudaCheckError(cudaFree(dest_dev));
  cudaCheckError(cudaFree(block_sums));

  // Compare with reference implementation
  std::unique_ptr<int[]> dest_reference(new int[COUNT]);
  scan_reference(source.get(), dest_reference.get(), COUNT);
  for (int i = 0; i < COUNT; i++) {
    assert(dest_reference.get()[i] == dest.get()[i]);
  }

  return 0;
}
