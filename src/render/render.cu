#include "camera.cuh"
#include "image.cuh"
#include "render.cuh"
#include "triangle.cuh"

#include <iostream>

Image render_normals(Triangle s, Camera cam, unsigned int w, unsigned int h) {
  Image out(w, h);
  TriangleNormals normals(s);

  for (unsigned int y = 0; y < h; y++) {
    for (unsigned int x = 0; x < w; x++) {
      float u = (float) x / (float) w;
      float v = (float) y / (float) h;

      Ray r = cam.get_ray(u, v);

      float3 rgb = {0, 0, 0};

      auto i = s.intersects(r);
      if (i.hit) {
        auto normal = normalized(normals.at(i.uvw, r));
        rgb.x       = (normal.x + 1.0) / 2.0;
        rgb.y       = (normal.y + 1.0) / 2.0;
        rgb.z       = (normal.z + 1.0) / 2.0;
      }

      out[y * w + x] = rgb;
    }
  }

  return out;
}
