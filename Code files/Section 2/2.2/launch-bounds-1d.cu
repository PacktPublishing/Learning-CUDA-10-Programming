// Demonstration of kernel execution configuration for a one-dimensional
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

__global__ void kernel_1d()
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  printf("block %d, thread %d, index %d\n", blockIdx.x, threadIdx.x, index);
}

int main()
{
  kernel_1d<<<4, 8>>>();
  cudaCheckError(cudaDeviceSynchronize());
}
