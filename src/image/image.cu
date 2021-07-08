#include "image.cuh"
#include "lodepng.h"

Image::Image(const std::string& filename) {
  unsigned char* pixels = nullptr;

  lodepng_decode32_file(&pixels, &width, &height, filename.c_str());

  float3* data = (float3*) malloc(width * height * sizeof(float3));
  for (unsigned int i = 0; i < width * height * 4; i += 4) {
    float r = (float) pixels[i + 0] / 255.0;
    float g = (float) pixels[i + 1] / 255.0;
    float b = (float) pixels[i + 2] / 255.0;
    // ignore alpha

    data[i / 4] = {r, g, b};
  }

  free(pixels);
}
