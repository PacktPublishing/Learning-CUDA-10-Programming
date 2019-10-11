// Matrix transpose using shared memory to ensure all writes coalesce.
// Example for video 4.1.

#include <assert.h>
#include <memory>

// Standard CUDA API functions
#include <cuda_runtime_api.h>

// CUDA cooperative groups API
#include <cooperative_groups.h>

#include "../utils.h"

const int TILE_DIM = 16;

// Reference implementation on the host
void transpose_reference(const float *source, float *dest,
                         unsigned int dimension)
{
  for (int y = 0; y < dimension; y++) {
    for (int x = 0; x < dimension; x++) {
      dest[y + x * dimension] = source[x + y * dimension];
    }
  }
}

// Transpose a matrix
// For simplicity, we assume that the matrix is square, and that its
// dimension is a multiple of the block size, so we don't have to worry about
// pitch or bounds checking.
__global__ void transpose(const float *source, float *dest,
                          unsigned int dimension)
{
  // Shared memory to temporarily store data.
  // Note the padding of the Y dimension, to avoid bank conflicts.
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int x_in = blockIdx.x * blockDim.x + threadIdx.x;
  int y_in = blockIdx.y * blockDim.y + threadIdx.y;
  int source_index = x_in + y_in * dimension;

  // Read from global memory to shared memory. Global memory access is
  // aligned.
  tile[threadIdx.y][threadIdx.x] = source[source_index];

  // Wait for all threads in the block to finish, so the shared memory tile
  // is filled.
  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();
  cooperative_groups::sync(block);

  // Output coordinates. Note that blockIdx.y is used to determine x_out, and
  // blockIdx.x is used to determine y_out.
  int x_out = blockIdx.y * blockDim.y + threadIdx.x;
  int y_out = blockIdx.x * blockDim.y + threadIdx.y;
  int dest_index = x_out + y_out * dimension;

  // Read from a different index in the shared memory tile, and write to
  // global memory. Global memory access is once again aligned.
  dest[dest_index] = tile[threadIdx.x][threadIdx.y];
}

int main(int argc, char **argv)
{
  const unsigned int DIMENSION = 4096;
  const unsigned int COUNT = DIMENSION * DIMENSION;
  std::unique_ptr<float[]> source(new float[COUNT]);
  std::unique_ptr<float[]> dest(new float[COUNT]);

  // Fill source matrix with some arbitrary test values
  for (int i = 0; i < COUNT; i++) {
    source[i] = i;
  }

  // Allocate and fill device memory
  float *source_dev, *dest_dev;
  size_t size = COUNT * sizeof(float);
  cudaCheckError(cudaMalloc(&dest_dev, size));
  cudaCheckError(cudaMalloc(&source_dev, size));
  cudaCheckError(
      cudaMemcpy(source_dev, source.get(), size, cudaMemcpyHostToDevice));

  // Run the kernel
  dim3 block_dim(TILE_DIM, TILE_DIM);
  dim3 grid_dim((DIMENSION + block_dim.x - 1) / block_dim.x,
                (DIMENSION + block_dim.y - 1) / block_dim.y);

  {
    KernelTimer t;
    transpose<<<grid_dim, block_dim>>>(source_dev, dest_dev, DIMENSION);
  }

  // Copy results back to the host
  cudaCheckError(
      cudaMemcpy(dest.get(), dest_dev, size, cudaMemcpyDeviceToHost));
  cudaCheckError(cudaFree(dest_dev));
  cudaCheckError(cudaFree(source_dev));

  // Compare with reference implementation
  std::unique_ptr<float[]> dest_reference(new float[COUNT]);
  transpose_reference(source.get(), dest_reference.get(), DIMENSION);

  for (int i = 0; i < COUNT; i++) {
    assert(dest_reference.get()[i] == dest.get()[i]);
  }

  return 0;
}
