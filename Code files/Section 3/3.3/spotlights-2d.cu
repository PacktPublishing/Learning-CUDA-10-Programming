// Render several spotlights on an image.
// Uses a two-dimensional memory layout to ensure coalesced access.
// Example for video 3.3.

#include <cmath>
#include <cstdint>
#include <iostream>

// Standard CUDA API functions
#include <cuda_runtime_api.h>

#include "../utils.h"

struct light {
  float x;
  float y;
  float radius;
  float brightness;
};

__device__ float clamp(float value) { return value > 1.0f ? 1.0f : value; }

__device__ float light_brightness(float x, float y, unsigned int width,
                                  unsigned int height, const light &light)
{
  float norm_x = x / width;
  float norm_y = y / height;

  float dx = norm_x - light.x;
  float dy = norm_y - light.y;
  float distance_squared = dx * dx + dy * dy;
  if (distance_squared > light.radius * light.radius) {
    return 0;
  }
  float distance = sqrtf(distance_squared);

  float scaled_distance = distance / light.radius;
  if (scaled_distance > 0.8) {
    return (1.0f - (scaled_distance - 0.8f) * 5.0f) * light.brightness;
  } else {
    return light.brightness;
  }
}

template <typename T>
__device__ T *pointer2d(T *base_pointer, int x, int y, size_t pitch)
{
  return (T *)((char *)base_pointer + y * pitch) + x;
}

__global__ void spotlights(const image source, image dest, unsigned int width,
                           unsigned int height, size_t pitch, float ambient,
                           light light1, light light2, light light3,
                           light light4)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  float brightness = ambient + light_brightness(x, y, width, height, light1) +
                     light_brightness(x, y, width, height, light2) +
                     light_brightness(x, y, width, height, light3) +
                     light_brightness(x, y, width, height, light4);

  *pointer2d(dest.red, x, y, pitch) =
      clamp(*pointer2d(source.red, x, y, pitch) * brightness);
  *pointer2d(dest.green, x, y, pitch) =
      clamp(*pointer2d(source.green, x, y, pitch) * brightness);
  *pointer2d(dest.blue, x, y, pitch) =
      clamp(*pointer2d(source.blue, x, y, pitch) * brightness);
}

int main(int argc, char **argv)
{
  auto params = set_up_test_planar(argc, argv);

  light light1 = {0.2, 0.1, 0.1, 4.0};
  light light2 = {0.25, 0.2, 0.075, 2.0};
  light light3 = {0.5, 0.5, 0.3, 0.3};
  light light4 = {0.7, 0.65, 0.15, 0.8};

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
  std::cout << "Width: " << byte_width << " bytes. Pitch: " << pitch
            << " bytes\n";

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

  dim3 BLOCK_DIM(32, 16);
  dim3 grid_dim((params.width + BLOCK_DIM.x - 1) / BLOCK_DIM.x,
                (params.height + BLOCK_DIM.y - 1) / BLOCK_DIM.y);

  {
    KernelTimer t;
    spotlights<<<grid_dim, BLOCK_DIM>>>(input2d, output2d, params.width,
                                        params.height, pitch, 0.3, light1,
                                        light2, light3, light4);
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
