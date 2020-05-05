#pragma once

#include "math.cuh"
#include "cuda.cuh"
#include "vector.cuh"

#include <string>

struct Texture {
  Vector<texture> data; // row-major
  uint32_t width;
  uint32_t height;

  Texture() : data(), width(0), height(0) {}
  Texture(const std::string &png_name);

  __device__ Vec3 eval(float u, float v) const;

  void to_host() { return data.to_host(); }
  void to_device() { return data.to_device(); }
};
