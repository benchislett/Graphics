#include "scene.h"

#include <cstdio>

unsigned char rescale(float x) {
  // printf("%f\n", x);
  x = fminf(fmaxf(0.f, x), 1.f);
  return (unsigned char) (x * 255.f);
}

void to_ppm(const Image image, const std::string& filename) {
  FILE* file = fopen(filename.c_str(), "wb");
  if (file == NULL) {
    fprintf(stderr, "Unable to write image to file %s\n", filename.c_str());
    return;
  }

  fprintf(file, "P6 %d %d %d\n", image.x, image.y, 255);

  int size = image.x * image.y;
  for (int i = 0; i < size; i++) {
    unsigned char rgb[3];
    rgb[0] = rescale(image.data[i].x);
    rgb[1] = rescale(image.data[i].y);
    rgb[2] = rescale(image.data[i].z);
    fwrite(rgb, 1, 3, file);
  }

  fclose(file);
}
