// Convert a color image to monochrome.
// Example for videos 3.1 and 3.2.

#include <cstdint>
#include <iostream>

// Standard CUDA API functions
#include <cuda_runtime_api.h>

#include "../utils.h"

__global__ void monochrome(const pixel *source, pixel *dest, int size)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) return;

  float value(source[index].red * 0.3125f + source[index].green * 0.5f +
              source[index].blue * .1875f);

  dest[index].red = value;
  dest[index].green = value;
  dest[index].blue = value;
  dest[index].alpha = source[index].alpha;
}

int main(int argc, char **argv)
{
  test_params params = set_up_test(argc, argv);

  int pixel_count = params.width * params.height;
  int BLOCK_SIZE = 128;
  int n_blocks = (pixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE;

  {
    KernelTimer t;
    monochrome<<<n_blocks, BLOCK_SIZE>>>(params.input_image,
                                         params.output_image, pixel_count);
  }

  finish_test(params);

  return 0;
}
