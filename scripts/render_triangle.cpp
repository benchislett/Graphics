#include "render.cuh"

int main() {
  Image out = render_triangle();
  out.to_png("../scripts/output/triangle.png");
  out.destroy();
  return 0;
}
