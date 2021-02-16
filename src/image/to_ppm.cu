#include "image.cuh"

#include <cstdio>

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
    rgb[0] = image.data[i].x;
    rgb[1] = image.data[i].y;
    rgb[2] = image.data[i].z;
    fwrite(rgb, 1, 3, file);
  }

  fclose(file);
}
