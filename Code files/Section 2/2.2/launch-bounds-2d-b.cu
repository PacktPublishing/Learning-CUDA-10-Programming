// Demonstration of kernel execution configuration for a two-dimensional
// grid.
// Example for video 2.2.

#include <cuda_runtime_api.h>
#include <stdio.h>

// Error checking macro
#define cudaCheckError(code)                                             \
  {                                                                      \
    if ((code) != cudaSuccess) {                                         \
      fprintf(stderr, "Cuda failure %s:%d: '%s' \n", __FILE__, __LINE__, \
              cudaGetErrorString(code));                                 \
    }                                                                    \
  }

__global__ void kernel_2d()
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  printf("block (%d, %d), thread (%d, %d), index (%d, %d)\n", blockIdx.x,
         blockIdx.y, threadIdx.x, threadIdx.y, x, y);
}

int main()
{
  dim3 block_dim(8, 2);
  dim3 grid_dim(2, 1);
  kernel_2d<<<grid_dim, block_dim>>>();
  cudaCheckError(cudaDeviceSynchronize());
}
