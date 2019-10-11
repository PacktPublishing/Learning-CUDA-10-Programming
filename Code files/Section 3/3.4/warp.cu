// Image warping using global memory.
// Reads are uncoalesced so performance is not optimal.
// Example for video 3.4.

#include <assert.h>
#include <cmath>
#include <cstdint>
#include <iostream>

// Standard CUDA API functions
#include <cuda_runtime_api.h>

#include "../utils.h"

struct warp_params {
  float matrix[4];
  float inverse_matrix[4];
  float x_shift;
  float y_shift;
};

template <typename T>
__device__ T *pointer2d(T *base_pointer, int x, int y, size_t pitch)
{
  return (T *)((char *)base_pointer + y * pitch) + x;
}

__device__ float get_pixel(const float *source, unsigned int width,
                           unsigned int height, size_t pitch, int x, int y)
{
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return 0.0f;
  } else {
    return *pointer2d(source, x, y, pitch);
  }
}

__device__ float average_pixels(const float *source, unsigned int width,
                                unsigned int height, size_t pitch, int x0,
                                float weight_x, int y0, float weight_y)

{
  float p00 = get_pixel(source, width, height, pitch, x0, y0);
  float p01 = get_pixel(source, width, height, pitch, x0, y0 + 1);
  float p10 = get_pixel(source, width, height, pitch, x0 + 1, y0);
  float p11 = get_pixel(source, width, height, pitch, x0 + 1, y0 + 1);

  return (p00 * weight_x + p10 * (1.0f - weight_x)) * weight_y +
         (p01 * weight_x + p11 * (1.0f - weight_x)) * (1.0f - weight_y);
}

__global__ void warp_image(const image source, image dest, unsigned int width,
                           unsigned int height, size_t pitch,
                           warp_params params)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  float source_x = params.inverse_matrix[0] * x + params.inverse_matrix[1] * y -
                   params.x_shift;
  float source_y = params.inverse_matrix[2] * x + params.inverse_matrix[3] * y -
                   params.y_shift;

  float x0 = floorf(source_x);
  float weight_x = source_x - x0;
  int x0_int = static_cast<int>(x0);
  float y0 = floorf(source_y);
  float weight_y = source_y - y0;
  int y0_int = static_cast<int>(y0);

  *pointer2d(dest.red, x, y, pitch) = average_pixels(
      source.red, width, height, pitch, x0_int, weight_x, y0_int, weight_y);
  *pointer2d(dest.green, x, y, pitch) = average_pixels(
      source.green, width, height, pitch, x0_int, weight_x, y0_int, weight_y);
  *pointer2d(dest.blue, x, y, pitch) = average_pixels(
      source.blue, width, height, pitch, x0_int, weight_x, y0_int, weight_y);
}

static void mult_matrix(float mat[4], float a, float b, float c, float d)
{
  float dst_a = mat[0] * a + mat[1] * c;
  float dst_b = mat[0] * b + mat[1] * d;
  float dst_c = mat[2] * a + mat[3] * c;
  float dst_d = mat[2] * b + mat[3] * d;

  mat[0] = dst_a;
  mat[1] = dst_b;
  mat[2] = dst_c;
  mat[3] = dst_d;
}

static void invert_matrix(float inverse[4], const float mat[4])
{
  float determinant = mat[0] * mat[3] - mat[1] * mat[2];
  assert(determinant != 0);  // Shouldn't happen if scales are non-zero
  float inverse_determinant = 1.0f / determinant;

  inverse[0] = mat[3] * inverse_determinant;
  inverse[1] = -1 * mat[1] * inverse_determinant;
  inverse[2] = -1 * mat[2] * inverse_determinant;
  inverse[3] = mat[0] * inverse_determinant;
}

int main(int argc, char **argv)
{
  auto params = set_up_test_planar(argc, argv);
  image input2d, output2d;
  size_t byte_width = params.width * sizeof(float);
  size_t pitch;

  // Allocate 2D aligned image
  cudaCheckError(
      cudaMallocPitch(&input2d.red, &pitch, byte_width, params.height));
  // Copy from 1D to 2D image
  cudaCheckError(cudaMemcpy2D(input2d.red, pitch, params.input_image.red,
                              byte_width, byte_width, params.height,
                              cudaMemcpyDeviceToDevice));

  // Allocate and copy other channels
  // Note: pitch will be the same for all of these allocations
  cudaCheckError(
      cudaMallocPitch(&input2d.green, &pitch, byte_width, params.height));
  cudaCheckError(
      cudaMallocPitch(&input2d.blue, &pitch, byte_width, params.height));
  cudaCheckError(
      cudaMallocPitch(&output2d.red, &pitch, byte_width, params.height));
  cudaCheckError(
      cudaMallocPitch(&output2d.green, &pitch, byte_width, params.height));
  cudaCheckError(
      cudaMallocPitch(&output2d.blue, &pitch, byte_width, params.height));
  cudaCheckError(cudaMemcpy2D(input2d.green, pitch, params.input_image.green,
                              byte_width, byte_width, params.height,
                              cudaMemcpyDeviceToDevice));
  cudaCheckError(cudaMemcpy2D(input2d.blue, pitch, params.input_image.blue,
                              byte_width, byte_width, params.height,
                              cudaMemcpyDeviceToDevice));

  // Set up warp parameters
  const float SCALE = 0.65f;
  const float ROTATE_RADS = 0.3;
  warp_params warp;
  // Scaling matrix
  warp.matrix[0] = warp.matrix[3] = SCALE;
  warp.matrix[1] = warp.matrix[2] = 0;
  // Add rotation
  mult_matrix(warp.matrix, cosf(ROTATE_RADS), sinf(ROTATE_RADS),
              -1 * sinf(ROTATE_RADS), cosf(ROTATE_RADS));
  // Kernel will use inverse
  invert_matrix(warp.inverse_matrix, warp.matrix);
  // Add translation
  warp.x_shift = 0.1f * params.width;
  warp.y_shift = 0.3f * params.height;

  dim3 BLOCK_DIM(32, 16);
  dim3 grid_dim((params.width + BLOCK_DIM.x - 1) / BLOCK_DIM.x,
                (params.height + BLOCK_DIM.y - 1) / BLOCK_DIM.y);

  {
    KernelTimer t;
    warp_image<<<grid_dim, BLOCK_DIM>>>(input2d, output2d, params.width,
                                        params.height, pitch, warp);
  }

  cudaCheckError(cudaMemcpy2D(params.output_image.red, byte_width, output2d.red,
                              pitch, byte_width, params.height,
                              cudaMemcpyDeviceToDevice));
  cudaCheckError(cudaMemcpy2D(params.output_image.green, byte_width,
                              output2d.green, pitch, byte_width, params.height,
                              cudaMemcpyDeviceToDevice));
  cudaCheckError(cudaMemcpy2D(params.output_image.blue, byte_width,
                              output2d.blue, pitch, byte_width, params.height,
                              cudaMemcpyDeviceToDevice));

  free_image(input2d);
  free_image(output2d);

  finish_test_planar(params);

  return 0;
}
