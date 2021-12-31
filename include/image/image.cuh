#pragma once

#include "alloc.cuh"
#include "float3.cuh"

#include <cstdlib>
#include <string>

struct Image {
  Vector<float3> values;
  unsigned int width;
  unsigned int height;

  Image() : width(0), height(0), values() {}
  Image(unsigned w, unsigned h) : width(w), height(h), values{w * h} {}
  Image(const std::string& filename);

  __host__ __device__ float3& operator[](int i) {
    return values[i];
  }

  void to_png(const std::string& filename) const;
};
