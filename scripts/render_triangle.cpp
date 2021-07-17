#include "camera.cuh"
#include "render.cuh"
#include "triangle.cuh"

#include <cmath>

int main() {
  Triangle t({2, -1, -1}, {2, 1, -1}, {2, 0, 1});
  Camera cam(M_PI / 4.0, 1.0, {-1, 0, 0}, {1, 0, 0});
  Image out = render_normals(t, cam, 32, 32);
  out.to_png("../scripts/output/triangle.png");
  out.destroy();
  return 0;
}
