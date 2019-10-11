// Demonstration of the unified virtual address space. Run multiple scans
// concurrently across all available devices
// Example for video 6.5.

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

const int BLOCK_SIZE = 1024;

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

static void print_pointer(const std::string &name, const void *pointer)
{
  cudaPointerAttributes attributes;
  auto result = cudaPointerGetAttributes(&attributes, pointer);

  std::cout << name << ": ";
  if (result != cudaSuccess) {
    std::cout << "get attributes failed";
    return;
  } else {
    switch (attributes.type) {
      case cudaMemoryTypeUnregistered:
        std::cout << "unregistered";
        break;
      case cudaMemoryTypeHost:
        std::cout << "host memory";
        break;
      case cudaMemoryTypeDevice:
        std::cout << "device " << attributes.device;
        break;
      case cudaMemoryTypeManaged:
        std::cout << "managed";
        break;
    }
  }

  std::cout << "\n";
}

int main(int argc, char **argv)
{
  // Maximum possible size with two-level scan.
  const unsigned int COUNT = BLOCK_SIZE * BLOCK_SIZE;
  const int N_STREAMS = 2;

  int *sources[N_STREAMS], *dests[N_STREAMS];

  // Fill source arrays with some arbitrary test values
  std::mt19937 rng;
  rng.seed(0);
  std::uniform_int_distribution<std::mt19937::result_type> dist(0, 9);

  int device_count;
  cudaCheckError(cudaGetDeviceCount(&device_count));

  for (int i = 0; i < N_STREAMS; i++) {
    // Allocate page-locked memory to allow asynchronous transfers.
    cudaMallocHost(&sources[i], COUNT * sizeof(int));
    cudaMallocHost(&dests[i], COUNT * sizeof(int));
    for (int j = 0; j < COUNT; j++) {
      sources[i][j] = dist(rng);
    }
  }

  // Allocate device memory and transfer data
  int n_blocks1 = (COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE;

  int *sources_dev[N_STREAMS], *dests_dev[N_STREAMS], *block_sums[N_STREAMS];
  size_t size = COUNT * sizeof(int);
  cudaStream_t stream[N_STREAMS];

  for (int i = 0; i < N_STREAMS; i++) {
    int device = i % device_count;
    cudaCheckError(cudaSetDevice(device));
    cudaCheckError(cudaStreamCreate(&stream[i]));
    cudaCheckError(cudaMalloc(&sources_dev[i], size));
    cudaCheckError(cudaMalloc(&dests_dev[i], size));
    // Temporary buffer for kernels
    cudaCheckError(cudaMalloc(&block_sums[i], n_blocks1 * sizeof(int)));
  }

  {
    KernelTimer t;

    for (int i = 0; i < N_STREAMS; i++) {
      int device = i % device_count;
      cudaCheckError(cudaSetDevice(device));

      std::cout << "Stream " << i << " on device " << device << "\n";
      print_pointer("source", sources[i]);
      print_pointer("source_dev", sources_dev[i]);
      print_pointer("dest_dev", dests_dev[i]);
      print_pointer("dest", dests[i]);

      // Copy data to device
      cudaCheckError(cudaMemcpyAsync(sources_dev[i], sources[i], size,
                                     cudaMemcpyDefault, stream[i]));

      // Run the scan
      scan1<<<n_blocks1, BLOCK_SIZE, 0, stream[i]>>>(sources_dev[i],
                                                     dests_dev[i]);

      int n_blocks2 = (n_blocks1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
      assert(n_blocks2 == 1);
      scan2<<<n_blocks2, BLOCK_SIZE, 0, stream[i]>>>(dests_dev[i],
                                                     block_sums[i], n_blocks1);

      finish_scan<<<n_blocks1, BLOCK_SIZE, 0, stream[i]>>>(block_sums[i],
                                                           dests_dev[i]);

      // Copy results back to the host
      cudaCheckError(cudaMemcpyAsync(dests[i], dests_dev[i], size,
                                     cudaMemcpyDefault, stream[i]));
      std::cout << "\n";
    }
  }

  for (int i = 0; i < N_STREAMS; i++) {
    cudaCheckError(cudaFree(sources_dev[i]));
    cudaCheckError(cudaFree(dests_dev[i]));
    cudaCheckError(cudaFree(block_sums[i]));
  }

  // Compare with reference implementation
  std::unique_ptr<int[]> dest_reference(new int[COUNT]);
  for (int i = 0; i < N_STREAMS; i++) {
    scan_reference(sources[i], dest_reference.get(), COUNT);
    for (int j = 0; j < COUNT; j++) {
      assert(dest_reference.get()[j] == dests[i][j]);
    }
  }

  return 0;
}
