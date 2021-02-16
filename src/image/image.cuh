#pragma once

#include "cu_math.cuh"

#include <string>

struct Image {
  int x;
  int y;
  uchar4* data;
};

void to_ppm(const Image image, const std::string& filename);
