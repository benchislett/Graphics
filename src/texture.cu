#include "texture.cuh"

#include "lodepng/lodepng.h"

Texture::Texture(const std::string &png_name) {
  std::vector<unsigned char> pixels;
  unsigned error = lodepng::decode(pixels, width, height, png_name.c_str());

  if (error) {
    fprintf(stderr, "Error loading texture from %s\n", png_name.c_str());
    data = NULL;
    return;
  }

  data = (Vec3 *)malloc(width * height * sizeof(Vec3));
  for (int i = 0; i < pixels.size(); i += 4) {
    data[i / 4] = {(float)pixels[i] / 255.f, (float)pixels[i + 1] / 255.f, (float)pixels[i + 2] / 255.f};
  }
}

Vec3 Texture::eval(float u, float v) const {
  int i = (int)(u * width);
  int j = (int)(v * height);
  return data[j * width + i];
}
