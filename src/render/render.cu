#include "camera.cuh"
#include "image.cuh"
#include "render.cuh"
#include "triangle.cuh"

#include <cmath>

Image render_triangle() {
  Image out(512, 512);
  Camera cam(M_PI / 3.0, 1.0, {-1, 0, 0}, {1, 0, 0});
  Triangle t({2, -1, -1}, {2, 1, -1}, {2, 0, 1});

  for (int y = 0; y < 512; y++) {
    for (int x = 0; x < 512; x++) {
      float u = (float) x / 512.0;
      float v = (float) y / 512.0;

      Ray r = cam.get_ray(u, v);

      float3 rgb = {0, 0, 0};

      auto i = t.intersects(r);
      if (i.hit)
        rgb = i.uvw;

      out[y * 512 + x] = rgb;
    }
  }

  return out;
}
