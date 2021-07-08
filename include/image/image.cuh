#pragma once

#include "../math/float3.cuh"

#include <cstdlib>
#include <string>

struct Image {
  float3* data;
  unsigned int width;
  unsigned int height;

  Image() : width(0), height(0), data(nullptr) {}
  Image(unsigned int w, unsigned int h, float3* d) : data(d), width(w), height(h) {}
  Image(unsigned int w, unsigned int h, float* d) : data((float3*) d), width(w), height(h) {}
  Image(const std::string& filename);

  void to_png(const std::string& filename) const;
};