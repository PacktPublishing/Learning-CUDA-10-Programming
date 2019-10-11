// Demonstration of basic CUDA error handling.
// Example fgor video 2.5.

#include <stdio.h>

// Standard CUDA API functions
#include <cuda_runtime_api.h>

// Error checking macro
#define cudaCheckError(code)                                             \
  {                                                                      \
    if ((code) != cudaSuccess) {                                         \
      fprintf(stderr, "Cuda failure %s:%d: '%s' \n", __FILE__, __LINE__, \
              cudaGetErrorString(code));                                 \
    }                                                                    \
  }

__global__ void bad()
{
  char *x = nullptr;
  *x = 1;
}

__global__ void good() {}

int main()
{
  int *foo = nullptr;
  size_t size = 1lu << 33;
  cudaError_t status = cudaMalloc(&foo, size);
  const char *message = cudaGetErrorString(status);

  status = cudaGetLastError();

  status = cudaMalloc(&foo, 16);
  message = cudaGetErrorString(status);

  bad<<<1, 1>>>();
  status = cudaDeviceSynchronize();
  message = cudaGetErrorString(status);

  good<<<1, 16>>>();
  status = cudaDeviceSynchronize();
  message = cudaGetErrorString(status);

  cudaCheckError(cudaMalloc(&foo, 16))

      return 0;
}
