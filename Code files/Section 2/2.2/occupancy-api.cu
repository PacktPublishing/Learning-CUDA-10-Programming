// Demonstration of the CUDA occupancy API.
// Example for video 2.2.

#include <cuda_runtime_api.h>
#include <stdio.h>

__global__ void kernel_1d() {}

int main()
{
  int block_size;     // The launch configurator returned block size
  int min_grid_size;  // The minimum grid size needed to achieve max occupancy

  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_1d, 0,
                                     0);

  printf("Block size %d\nMin grid size %d\n", block_size, min_grid_size);
}
