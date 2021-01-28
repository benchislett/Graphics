#include "integrate.h"

Image render(const Scene scene, int x, int y, int spp) {
  Image image;
  image.x    = x;
  image.y    = y;
  image.data = (float3*) malloc(x * y * sizeof(float3));

  for (int i = 0; i < x * y; i++) {
    float3 val = make_float3(0.f);
    for (int s = 0; s < spp; s++) {
      val += make_float3(0.5f);
    }
    image.data[i] = val / make_float3((float) spp);
  }

  return image;
}
