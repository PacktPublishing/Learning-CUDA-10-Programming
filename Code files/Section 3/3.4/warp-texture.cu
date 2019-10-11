// Image warping using texture memory to improve performance.
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

struct texture_image {
  cudaTextureObject_t red;
  cudaTextureObject_t green;
  cudaTextureObject_t blue;
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

__global__ void warp_image(texture_image source, image dest, unsigned int width,
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

  *pointer2d(dest.red, x, y, pitch) =
      tex2D<float>(source.red, source_x, source_y);
  *pointer2d(dest.green, x, y, pitch) =
      tex2D<float>(source.green, source_x, source_y);
  *pointer2d(dest.blue, x, y, pitch) =
      tex2D<float>(source.blue, source_x, source_y);
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
  image output2d;
  size_t byte_width = params.width * sizeof(float);
  size_t pitch;

  cudaCheckError(
      cudaMallocPitch(&output2d.red, &pitch, byte_width, params.height));
  cudaCheckError(
      cudaMallocPitch(&output2d.green, &pitch, byte_width, params.height));
  cudaCheckError(
      cudaMallocPitch(&output2d.blue, &pitch, byte_width, params.height));

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

  // Create arrays: opaque memory layouts optimized for texture
  // fetching. Copy our input images to them.
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaArray *red_array, *green_array, *blue_array;
  cudaCheckError(
      cudaMallocArray(&red_array, &channelDesc, params.width, params.height));
  cudaCheckError(cudaMemcpy2DToArray(
      red_array, 0, 0, params.input_image.red, params.width * sizeof(float),
      params.width * sizeof(float), params.height, cudaMemcpyDeviceToDevice));
  cudaCheckError(
      cudaMallocArray(&green_array, &channelDesc, params.width, params.height));
  cudaCheckError(cudaMemcpy2DToArray(
      green_array, 0, 0, params.input_image.green, params.width * sizeof(float),
      params.width * sizeof(float), params.height, cudaMemcpyDeviceToDevice));
  cudaCheckError(
      cudaMallocArray(&blue_array, &channelDesc, params.width, params.height));
  cudaCheckError(cudaMemcpy2DToArray(
      blue_array, 0, 0, params.input_image.blue, params.width * sizeof(float),
      params.width * sizeof(float), params.height, cudaMemcpyDeviceToDevice));

  // Create resource descriptions for each channel, for use in texture setup.
  struct cudaResourceDesc red_resource = {cudaResourceTypeArray};
  red_resource.res.array.array = red_array;
  struct cudaResourceDesc green_resource = {cudaResourceTypeArray};
  green_resource.res.array.array = green_array;
  struct cudaResourceDesc blue_resource = {cudaResourceTypeArray};
  blue_resource.res.array.array = blue_array;

  // Create texture description, specifying settings for texture fetches.
  struct cudaTextureDesc texture_desc = {};
  texture_desc.addressMode[0] = cudaAddressModeBorder;
  texture_desc.addressMode[1] = cudaAddressModeBorder;
  texture_desc.filterMode = cudaFilterModeLinear;
  texture_desc.readMode = cudaReadModeElementType;
  texture_desc.normalizedCoords = 0;

  // Create texture objects which combine the resources and the texture
  // descriptions.
  texture_image source_texture;
  cudaCreateTextureObject(&source_texture.red, &red_resource, &texture_desc,
                          NULL);
  cudaCreateTextureObject(&source_texture.green, &green_resource, &texture_desc,
                          NULL);
  cudaCreateTextureObject(&source_texture.blue, &blue_resource, &texture_desc,
                          NULL);

  dim3 BLOCK_DIM(32, 16);
  dim3 grid_dim((params.width + BLOCK_DIM.x - 1) / BLOCK_DIM.x,
                (params.height + BLOCK_DIM.y - 1) / BLOCK_DIM.y);

  {
    KernelTimer t;
    warp_image<<<grid_dim, BLOCK_DIM>>>(source_texture, output2d, params.width,
                                        params.height, pitch, warp);
  }

  cudaCheckError(cudaDestroyTextureObject(source_texture.red));
  cudaCheckError(cudaDestroyTextureObject(source_texture.green));
  cudaCheckError(cudaDestroyTextureObject(source_texture.blue));

  cudaCheckError(cudaFreeArray(red_array));
  cudaCheckError(cudaFreeArray(green_array));
  cudaCheckError(cudaFreeArray(blue_array));

  cudaCheckError(cudaMemcpy2D(params.output_image.red, byte_width, output2d.red,
                              pitch, byte_width, params.height,
                              cudaMemcpyDeviceToDevice));
  cudaCheckError(cudaMemcpy2D(params.output_image.green, byte_width,
                              output2d.green, pitch, byte_width, params.height,
                              cudaMemcpyDeviceToDevice));
  cudaCheckError(cudaMemcpy2D(params.output_image.blue, byte_width,
                              output2d.blue, pitch, byte_width, params.height,
                              cudaMemcpyDeviceToDevice));

  free_image(output2d);

  finish_test_planar(params);

  return 0;
}
