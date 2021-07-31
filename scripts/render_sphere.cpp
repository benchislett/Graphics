#include "camera.cuh"
#include "render.cuh"
#include "triangle.cuh"
#include "trimesh.cuh"

#include <cmath>
#include <vector>

int main() {
  std::vector<Triangle> tris;
  tris.emplace_back((float3){2, -1, -1}, (float3){2, 1, -1}, (float3){2, 0, 1});
  TriMesh mesh(tris.data(), tris.size());
  Camera cam(M_PI / 4.0, 1.0, {-1, 0, 0}, {1, 0, 0});
  Image out = render_normals(mesh, cam, 128, 128);
  out.to_png("../scripts/output/sphere.png");
  out.destroy();
  return 0;
}
