// Example of generating two-dimensional data coordinates from a
// one-dimensional grid. A two-dimensional grid would be better suited here.
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

__global__ void kernel_1d(int width)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int x = index % width;
  int y = index / width;
  printf("block %d, thread %d, index (%d, %d)\n", blockIdx.x, threadIdx.x, x,
         y);
}

int main()
{
  kernel_1d<<<4, 8>>>(16);
  cudaCheckError(cudaDeviceSynchronize());
}
