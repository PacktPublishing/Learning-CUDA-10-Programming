// Utility functions for example programs.

#ifndef __UTILS_H
#define __UTILS_H

#include <chrono>

// Error checking macro
#define cudaCheckError(code)                                             \
  {                                                                      \
    if ((code) != cudaSuccess) {                                         \
      fprintf(stderr, "Cuda failure %s:%d: '%s' \n", __FILE__, __LINE__, \
              cudaGetErrorString(code));                                 \
    }                                                                    \
  }

/* A single pixel with floating-point channel values */
struct pixel {
  float red;
  float green;
  float blue;
  float alpha;
};

/* An image with planar layout: separate buffers for each color channel */
struct image {
  float *red;
  float *green;
  float *blue;
};

bool loadPPM(const char *file, pixel **data, unsigned int *w, unsigned int *h);
void savePPM(const char *file, pixel *data, unsigned int w, unsigned int h);

struct test_params {
  unsigned int width;
  unsigned int height;
  /* Device pointers to images */
  pixel *input_image;
  pixel *output_image;
  const char *output_file;
};

struct test_params_planar {
  unsigned int width;
  unsigned int height;
  /* Device pointers to images */
  image input_image;
  image output_image;
  const char *output_file;
};

test_params set_up_test(int argc, char **argv);
void finish_test(const test_params &params);
test_params_planar set_up_test_planar(int argc, char **argv);
void finish_test_planar(const test_params_planar &params);
void free_image(const image &img);

class KernelTimer
{
 public:
  KernelTimer();
  ~KernelTimer();

 private:
  std::chrono::time_point<std::chrono::steady_clock> start;
};

#endif  // __UTILS_H
