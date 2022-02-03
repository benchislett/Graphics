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
  auto scene = load_obj("../scripts/scenes/sportsCar.obj");

  Camera cam(M_PI / 4.0, 1.0, {1, 2.25, 4}, {-0.5, 0, 0}); // Sports Car
  // Camera cam(M_PI / 4.0, 1.0, {0, 150, 175}, {0, 0, 0}); // Teapot

  Image out = render_normals(scene.primitives, cam, 128, 128);
  out.to_png("../scripts/output/output.png");
  return 0;
}
