#pragma once

#include "math.cuh"
#include "cuda.cuh"

#include <string>

struct Texture {
  Vec3 *data; // row-major
  uint32_t width;
  uint32_t height;

  Texture() : data(NULL), width(0), height(0) {}
  Texture(const std::string &png_name);

  __device__ Vec3 eval(float u, float v) const;
};
