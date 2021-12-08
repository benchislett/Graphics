#include "camera.cuh"
#include "render.cuh"
#include "triangle.cuh"
#include "trimesh.cuh"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

int main() {
  Triangle t((float3){2, -1, -1}, (float3){2, 1, -1}, (float3){2, 0, 1});
  Camera cam(M_PI / 4.0, 1.0, {-1, 0, 0}, {1, 0, 0});
  Image out = render_normals(t, cam, 4096, 4096);
  out.to_png("../scripts/output/output.png");
  return 0;
}
