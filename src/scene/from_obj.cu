#include "scene.cuh"

#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

Scene from_obj(const std::string& filename) {
  static std::vector<float3> vertices;
  static std::vector<float3> normals;
  static std::vector<Triangle> triangles;
  static std::vector<TriangleNormal> triangle_normals;
  static std::vector<TriangleEmissivity> triangle_emissivities;
  static std::vector<int> lights;

  TriangleEmissivity current_emit;
  current_emit.intensity = make_float3(0.f);

  std::ifstream file;
  file.open(filename, std::ios::in);

  if (!file) {
    std::cout << "Error opening file " << filename << '\n';
    exit(0);
  }

  std::string token;
  for (std::string line; std::getline(file, line);) {
    std::stringstream tokens(line);
    tokens >> token;

    if (token == "v") {
      float3 vertex;
      tokens >> vertex.x >> vertex.y >> vertex.z;
      vertices.push_back(vertex);
    } else if (token == "vn") {
      float3 normal;
      tokens >> normal.x >> normal.y >> normal.z;
      normals.push_back(normal);
    } else if (token == "f") {
      int v[3], n[3];
      for (int i = 0; i < 3; i++) {
        tokens >> token;
        sscanf(token.c_str(), "%d//%d", &v[i], &n[i]);
        v[i]--;
        n[i]--;
      }
      triangles.push_back((Triangle){vertices[v[0]], vertices[v[1]], vertices[v[2]]});
      triangle_normals.push_back((TriangleNormal){normals[n[0]], normals[n[1]], normals[n[2]]});
      triangle_emissivities.push_back(current_emit);
      if (length(current_emit.intensity) > 0.f) {
        lights.push_back(triangles.size() - 1);
      }
    } else if (token == "usemtl") {
      tokens >> token;
      if (token == "Light") {
        current_emit.intensity = make_float3(1.f);
      } else {
        current_emit.intensity = make_float3(0.f);
      }
    } else {
      std::cout << "Skipping line with unrecognized identifier " << token << '\n';
    }
  }

  Scene scene;
  scene.n_triangles  = triangles.size();
  scene.triangles    = triangles.data();
  scene.normals      = triangle_normals.data();
  scene.emissivities = triangle_emissivities.data();
  scene.n_lights     = lights.size();
  scene.lights       = lights.data();

  return scene;
}
