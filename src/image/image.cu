#include "image.cuh"
#include "lodepng.h"

#include <iostream>
#include <vector>

Image::Image(const std::string& filename) {
  std::vector<unsigned char> pixels;

  unsigned error = lodepng::decode(pixels, width, height, filename);
  if (error) {
    std::cerr << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
  }


  data = (float3*) malloc(width * height * sizeof(float3));
  for (unsigned int i = 0; i < width * height * 4; i += 4) {
    float r = ((float) pixels[i + 0]) / 255.0;
    float g = ((float) pixels[i + 1]) / 255.0;
    float b = ((float) pixels[i + 2]) / 255.0;
    // ignore alpha

    data[i / 4] = {r, g, b};
  }
}

void Image::destroy() {
  free(data);
  data = nullptr;
}

void Image::to_png(const std::string& filename) const {
  std::vector<unsigned char> pixels(width * height * 4);

  for (unsigned int i = 0; i < width * height * 4; i += 4) {
    float3 rgb    = data[i / 4];
    pixels[i + 0] = (unsigned char) (255.0 * rgb.x);
    pixels[i + 1] = (unsigned char) (255.0 * rgb.y);
    pixels[i + 2] = (unsigned char) (255.0 * rgb.z);
    pixels[i + 3] = 255;
  }

  unsigned error = lodepng::encode(filename, pixels, width, height);
  if (error) {
    std::cerr << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
  }
}
