// Concurrent execution of multiple single-block reductions.
// The reduce kernel is very efficient, but occupancy is low, so multiple
// concurrent launches are needed to achieve good throughput.
// Example for video 6.1.

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

__device__ int reduce_block(int value, int sdata[],
                            cooperative_groups::thread_block block)
{
  auto tid = threadIdx.x;

  // Fill shared memory with initial values
  sdata[tid] = value;

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

// Sum the source array and store the sum in dest.
// Requires block_size * sizeof(int) bytes of shared memory.

// This kernel should always be launched with a single block. Unlike the
// previous reduce example, it keeps all threads busy and does not store any
// temporary data in global memory. However, occupancy is very low due to
// running a single block.
__global__ void reduce_single_block(const int *source, int *dest,
                                    unsigned int count)
{
  extern __shared__ int sdata[];

  int sum = 0;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    sum += source[i];
  }

  sum = reduce_block(sum, sdata, cooperative_groups::this_thread_block());

  // The last thread of the block writes the result into global memory
  if (threadIdx.x == 0) {
    *dest = sum;
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

  int N_STREAMS = 16;
  int *results[N_STREAMS];
  int *sources[N_STREAMS];
  cudaStream_t stream[N_STREAMS];

  // Create streams, and allocate input and output for each stream.
  size_t size = COUNT * sizeof(int);
  for (int i = 0; i < N_STREAMS; i++) {
    cudaCheckError(cudaStreamCreate(&stream[i]));
    cudaCheckError(cudaMalloc(&results[i], sizeof(int)));
    cudaCheckError(cudaMalloc(&sources[i], size));
    cudaCheckError(
        cudaMemcpy(sources[i], source.get(), size, cudaMemcpyHostToDevice));
  }

  // Run the kernel
  const int BLOCK_SIZE = 256;
  size_t shared_memory_size = BLOCK_SIZE * sizeof(int);

  {
    KernelTimer t;
    for (int i = 0; i < N_STREAMS; i++) {
      // Launch each instance of this kernel in a separate stream.
      reduce_single_block<<<1, BLOCK_SIZE, shared_memory_size, stream[i]>>>(
          sources[i], results[i], COUNT);
    }

    // All work has been dispatched to the device. The kernels will run
    // concurrently if there is room on the device. The host id idle now, and
    // we can do additional concurrent processing on the host.
  }

  // Wait for all streams to finish
  cudaCheckError(cudaDeviceSynchronize());

  // Copy result back to the host
  int result;
  cudaCheckError(
      cudaMemcpy(&result, results[0], sizeof(result), cudaMemcpyDeviceToHost));
  for (int i = 0; i < N_STREAMS; i++) {
    cudaCheckError(cudaFree(sources[i]));
    cudaCheckError(cudaFree(results[i]));
  }

  // Compare with reference implementation
  int result_reference = std::accumulate(source.get(), source.get() + COUNT, 0);
  std::cout << "Sum of " << COUNT << " elements: " << result << "\n";
  assert(result_reference == result);

  return 0;
}
