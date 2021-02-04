#include "integrate.h"

#include <iostream>

float3 trace(const Ray ray, const Scene scene) {
  TriangleHitRecord record = first_hit(ray, scene.triangles, scene.n_triangles);
  if (record.hit) {
    return make_float3(record.u, record.v, 1.f - record.u - record.v);
  } else {
    return make_float3(0.f);
  }
}

Image render(const Camera camera, const Scene scene, int x, int y, int spp) {
  Image image;
  image.x    = x;
  image.y    = y;
  image.data = (uchar4*) malloc(x * y * sizeof(uchar4));

  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      float3 val = make_float3(0.f);
      float u    = (float) i / (float) x;
      float v    = (float) j / (float) y;
      for (int s = 0; s < spp; s++) {
        Ray ray = get_ray(camera, u, v);
        val += trace(ray, scene);
      }
      val /= (float) spp;
      val = clamp(val, 0.f, 1.f);
      val *= 255.9999f;
      uchar4 data;
      data.x = (unsigned char) val.x;
      data.y = (unsigned char) val.y;
      data.z = (unsigned char) val.z;
      data.w = (unsigned char) 255;

      image.data[i * y + j] = data;
    }
  }

  return image;
}
