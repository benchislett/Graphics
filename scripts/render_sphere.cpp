#include "camera.cuh"
#include "render.cuh"
#include "triangle.cuh"

#include <cmath>

int main() {
  Sphere s({2, 0, 0}, 0.5);
  Camera cam(M_PI / 4.0, 1.0, {-1, 0, 0}, {1, 0, 0});
  Image out = render_normals(s, cam, 512, 512);
  out.to_png("../scripts/output/sphere.png");
  out.destroy();
  return 0;
}
