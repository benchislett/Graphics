#include "camera.cuh"
#include "obj.cuh"
#include "render.cuh"
#include "tri_array.cuh"
#include "triangle.cuh"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

int main() {
  auto scene = load_obj("../scripts/scenes/CornellBox-Original.obj");

  Camera cam(M_PI / 4.0, 1.0, {0, 5, 10}, {0, 2, 0});
  Image out = render_normals(scene.primitives, cam, 512, 512);
  out.to_png("../scripts/output/output.png");
  return 0;
}
