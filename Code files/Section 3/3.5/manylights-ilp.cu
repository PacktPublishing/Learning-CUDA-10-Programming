// Render many spotlights on an image, computing multiple results per thread
// in order to increase instruction-level parallelism.
// Example for video 3.5.

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

struct lots_of_lights {
  unsigned int count;
  light lights[1024];
};

__constant__ lots_of_lights dev_lights;

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

const int OUTPUTS_PER_THREAD = 2;

__global__ void spotlights(const image source, image dest, unsigned int width,
                           unsigned int height, size_t pitch, float ambient)
{
  for (int i = 0; i < OUTPUTS_PER_THREAD; i++) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = OUTPUTS_PER_THREAD * blockIdx.y * blockDim.y + threadIdx.y +
            i * blockDim.y;
    if (x >= width || y >= height) return;

    float brightness = ambient;
    for (int i = 0; i < dev_lights.count; i++) {
      brightness += light_brightness(x, y, width, height, dev_lights.lights[i]);
    }

    *pointer2d(dest.red, x, y, pitch) =
        clamp(*pointer2d(source.red, x, y, pitch) * brightness);
    *pointer2d(dest.green, x, y, pitch) =
        clamp(*pointer2d(source.green, x, y, pitch) * brightness);
    *pointer2d(dest.blue, x, y, pitch) =
        clamp(*pointer2d(source.blue, x, y, pitch) * brightness);
  }
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

  lots_of_lights lights = {1024};
  float spacing = 1.0f / 32.0f;
  for (int x = 0; x < 32; x++) {
    for (int y = 0; y < 32; y++) {
      int index = y * 32 + x;
      lights.lights[index] = {x * spacing, y * spacing, 0.05, 0.2};
    }
  }

  cudaCheckError(
      cudaMemcpyToSymbol(dev_lights, &lights, sizeof(lots_of_lights)));

  dim3 BLOCK_DIM(32, 16);
  dim3 grid_dim(
      (params.width + BLOCK_DIM.x - 1) / BLOCK_DIM.x,
      (params.height + BLOCK_DIM.y - 1) / (BLOCK_DIM.y * OUTPUTS_PER_THREAD));

  {
    KernelTimer t;
    spotlights<<<grid_dim, BLOCK_DIM>>>(input2d, output2d, params.width,
                                        params.height, pitch, 0.0);
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
