#include "camera.cuh"
#include "render.cuh"
#include "tri_array.cuh"
#include "triangle.cuh"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

int main() {
  Vector<Vec3> vertices(20);
  vertices[0]  = Vec3(-0.57735, -0.57735, 0.57735);
  vertices[1]  = Vec3(0.934172, 0.356822, 0);
  vertices[2]  = Vec3(0.934172, -0.356822, 0);
  vertices[3]  = Vec3(-0.934172, 0.356822, 0);
  vertices[4]  = Vec3(-0.934172, -0.356822, 0);
  vertices[5]  = Vec3(0, 0.934172, 0.356822);
  vertices[6]  = Vec3(0, 0.934172, -0.356822);
  vertices[7]  = Vec3(0.356822, 0, -0.934172);
  vertices[8]  = Vec3(-0.356822, 0, -0.934172);
  vertices[9]  = Vec3(0, -0.934172, -0.356822);
  vertices[10] = Vec3(0, -0.934172, 0.356822);
  vertices[11] = Vec3(0.356822, 0, 0.934172);
  vertices[12] = Vec3(-0.356822, 0, 0.934172);
  vertices[13] = Vec3(0.57735, 0.57735, -0.57735);
  vertices[14] = Vec3(0.57735, 0.57735, 0.57735);
  vertices[15] = Vec3(-0.57735, 0.57735, -0.57735);
  vertices[16] = Vec3(-0.57735, 0.57735, 0.57735);
  vertices[17] = Vec3(0.57735, -0.57735, -0.57735);
  vertices[18] = Vec3(0.57735, -0.57735, 0.57735);
  vertices[19] = Vec3(-0.57735, -0.57735, -0.57735);

  TriangleArray tris(36);

  tris[0]  = Triangle(vertices[19 - 1], vertices[3 - 1], vertices[2 - 1]);
  tris[1]  = Triangle(vertices[12 - 1], vertices[19 - 1], vertices[2 - 1]);
  tris[2]  = Triangle(vertices[15 - 1], vertices[12 - 1], vertices[2 - 1]);
  tris[3]  = Triangle(vertices[8 - 1], vertices[14 - 1], vertices[2 - 1]);
  tris[4]  = Triangle(vertices[18 - 1], vertices[8 - 1], vertices[2 - 1]);
  tris[5]  = Triangle(vertices[3 - 1], vertices[18 - 1], vertices[2 - 1]);
  tris[6]  = Triangle(vertices[20 - 1], vertices[5 - 1], vertices[4 - 1]);
  tris[7]  = Triangle(vertices[9 - 1], vertices[20 - 1], vertices[4 - 1]);
  tris[8]  = Triangle(vertices[16 - 1], vertices[9 - 1], vertices[4 - 1]);
  tris[9]  = Triangle(vertices[13 - 1], vertices[17 - 1], vertices[4 - 1]);
  tris[10] = Triangle(vertices[1 - 1], vertices[13 - 1], vertices[4 - 1]);
  tris[11] = Triangle(vertices[5 - 1], vertices[1 - 1], vertices[4 - 1]);
  tris[12] = Triangle(vertices[7 - 1], vertices[16 - 1], vertices[4 - 1]);
  tris[13] = Triangle(vertices[6 - 1], vertices[7 - 1], vertices[4 - 1]);
  tris[14] = Triangle(vertices[17 - 1], vertices[6 - 1], vertices[4 - 1]);
  tris[15] = Triangle(vertices[6 - 1], vertices[15 - 1], vertices[2 - 1]);
  tris[16] = Triangle(vertices[7 - 1], vertices[6 - 1], vertices[2 - 1]);
  tris[17] = Triangle(vertices[14 - 1], vertices[7 - 1], vertices[2 - 1]);
  tris[18] = Triangle(vertices[10 - 1], vertices[18 - 1], vertices[3 - 1]);
  tris[19] = Triangle(vertices[11 - 1], vertices[10 - 1], vertices[3 - 1]);
  tris[20] = Triangle(vertices[19 - 1], vertices[11 - 1], vertices[3 - 1]);
  tris[21] = Triangle(vertices[11 - 1], vertices[1 - 1], vertices[5 - 1]);
  tris[22] = Triangle(vertices[10 - 1], vertices[11 - 1], vertices[5 - 1]);
  tris[23] = Triangle(vertices[20 - 1], vertices[10 - 1], vertices[5 - 1]);
  tris[24] = Triangle(vertices[20 - 1], vertices[9 - 1], vertices[8 - 1]);
  tris[25] = Triangle(vertices[10 - 1], vertices[20 - 1], vertices[8 - 1]);
  tris[26] = Triangle(vertices[18 - 1], vertices[10 - 1], vertices[8 - 1]);
  tris[27] = Triangle(vertices[9 - 1], vertices[16 - 1], vertices[7 - 1]);
  tris[28] = Triangle(vertices[8 - 1], vertices[9 - 1], vertices[7 - 1]);
  tris[29] = Triangle(vertices[14 - 1], vertices[8 - 1], vertices[7 - 1]);
  tris[30] = Triangle(vertices[12 - 1], vertices[15 - 1], vertices[6 - 1]);
  tris[31] = Triangle(vertices[13 - 1], vertices[12 - 1], vertices[6 - 1]);
  tris[32] = Triangle(vertices[17 - 1], vertices[13 - 1], vertices[6 - 1]);
  tris[33] = Triangle(vertices[13 - 1], vertices[1 - 1], vertices[11 - 1]);
  tris[34] = Triangle(vertices[12 - 1], vertices[13 - 1], vertices[11 - 1]);
  tris[35] = Triangle(vertices[19 - 1], vertices[12 - 1], vertices[11 - 1]);

  Vector<TriangleNormals> normals_arr(36);
  for (int i = 0; i < 36; i++)
    normals_arr[i] = TriangleNormals(tris[i]);

  Camera cam(M_PI / 4.0, 1.0, {-1.8, -2, -3}, {0, 0, 0});
  Image out = render_normals(tris, normals_arr, cam, 512, 512);
  out.to_png("../scripts/output/output.png");
  return 0;
}
