// Add two arrays using CUDA.
/// Example for videos 1.5 and 2.1

#include <assert.h>
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

// Host function for array addition
void add_loop(float *dest, int n_elts, const float *a, const float *b)
{
  for (int i = 0; i < n_elts; i++) {
    dest[i] = a[i] + b[i];
  }
}

// Device kernel for array addition.
__global__ void add_kernel(float *dest, int n_elts, const float *a,
                           const float *b)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n_elts) return;

  dest[index] = a[index] + b[index];
}

int main()
{
  const int ARRAY_LENGTH = 100;

  // Generate some data on the host
  float host_array_a[ARRAY_LENGTH];
  float host_array_b[ARRAY_LENGTH];
  float host_array_dest[ARRAY_LENGTH];

  for (int i = 0; i < ARRAY_LENGTH; i++) {
    host_array_a[i] = 2 * i;
    host_array_b[i] = 2 * i + 1;
  }

  // Allocate device memory
  float *device_array_a, *device_array_b, *device_array_dest;
  cudaCheckError(cudaMalloc(&device_array_a, sizeof(host_array_a)));
  cudaCheckError(cudaMalloc(&device_array_b, sizeof(host_array_b)));
  cudaCheckError(cudaMalloc(&device_array_dest, sizeof(host_array_dest)));

  // Transfer data to device
  cudaCheckError(cudaMemcpy(device_array_a, host_array_a, sizeof(host_array_a),
                            cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(device_array_b, host_array_b, sizeof(host_array_b),
                            cudaMemcpyHostToDevice));

  // Calculate lauch configuration
  const int BLOCK_SIZE = 128;
  int n_blocks = (ARRAY_LENGTH + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Add arrays on device
  add_kernel<<<BLOCK_SIZE, n_blocks>>>(device_array_dest, ARRAY_LENGTH,
                                       device_array_a, device_array_b);

  // Meanwhile, add arrays on the host, for comparison
  add_loop(host_array_dest, ARRAY_LENGTH, host_array_a, host_array_b);

  // Copy result back to host and compare
  float host_array_tmp[ARRAY_LENGTH];
  cudaCheckError(cudaMemcpy(host_array_tmp, device_array_dest,
                            sizeof(host_array_tmp), cudaMemcpyDeviceToHost));
  for (int i = 0; i < ARRAY_LENGTH; i++) {
    assert(host_array_tmp[i] == host_array_dest[i]);
    printf("%g + %g = %g\n", host_array_a[i], host_array_b[i],
           host_array_tmp[i]);
  }

  return 0;
}
