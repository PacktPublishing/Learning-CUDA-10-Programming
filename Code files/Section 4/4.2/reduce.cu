// Reduce an array to a single value by summing all of its elements.
// Example for video 4.1.

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

__device__ unsigned int blocks_finished = 0;
// Wait for all blocks in the grid to execute this function.
// Returns true for thread 0 of the last block, false for all
// other threads.
__device__ bool wait_for_all_blocks()
{
  // Wait until global write is visible to all other blocks
  __threadfence();

  // Wait for all blocks to finish by atomically incrementing a counter
  bool is_last = false;
  if (threadIdx.x == 0) {
    unsigned int ticket = atomicInc(&blocks_finished, gridDim.x);
    is_last = (ticket == gridDim.x - 1);
  }
  if (is_last) {
    blocks_finished = 0;
  }
  return is_last;
}

__device__ int reduce_block(const int *source, int sdata[],
                            cooperative_groups::thread_block block)
{
  unsigned int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  auto tid = threadIdx.x;

  // Add two elements into shared memory
  sdata[tid] = source[index] + source[index + blockDim.x];

  cooperative_groups::sync(block);

  // When shared memory block is filled, reduce within that block.
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    int index = 2 * stride * tid;
    if (index < blockDim.x) {
      sdata[index] += sdata[index + stride];
    }
    cooperative_groups::sync(block);
  }

  return sdata[0];
}

// Sum the source array. The dest array must have one element per block --
// the first element will contain the final result, and the rest are used for
// temporary storage.
__global__ void reduce(const int *source, int *dest)
{
  extern __shared__ int sdata[];

  int block_result =
      reduce_block(source, sdata, cooperative_groups::this_thread_block());

  // The last thread of each block writes the block result into global memory
  if (threadIdx.x == 0) {
    dest[blockIdx.x] = block_result;
  }

  bool is_last = wait_for_all_blocks();

  // All blocks have passed the threadfence, so all writes are visible to all
  // blocks. Now we can use one thread to sum the results from each block.
  if (is_last) {
    int sum = 0;
    for (int i = 0; i < gridDim.x; i++) {
      sum += dest[i];
    }
    // Final sum goes in dest[0]
    dest[0] = sum;
  }
}

int main(int argc, char **argv)
{
  const unsigned int COUNT = 4096 * 4096;
  std::unique_ptr<int[]> source(new int[COUNT]);

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

  // Run the kernel
  int BLOCK_SIZE = 128;
  int n_blocks = (COUNT + BLOCK_SIZE - 1) / (2 * BLOCK_SIZE);

  cudaCheckError(cudaMalloc(&dest_dev, n_blocks * sizeof(int)));

  {
    KernelTimer t;
    size_t shared_memory_size = BLOCK_SIZE * sizeof(int);
    reduce<<<n_blocks, BLOCK_SIZE, shared_memory_size>>>(source_dev, dest_dev);
  }

  // Copy result back to the host
  int result;
  cudaCheckError(
      cudaMemcpy(&result, dest_dev, sizeof(result), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaFree(source_dev));
  cudaCheckError(cudaFree(dest_dev));

  // Compare with reference implementation
  int result_reference = std::accumulate(source.get(), source.get() + COUNT, 0);
  std::cout << "Sum of " << COUNT << " elements: " << result << "\n";
  assert(result_reference == result);

  return 0;
}
