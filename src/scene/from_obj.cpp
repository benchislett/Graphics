#include "log.h"
#include "scene.h"

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
        v[i] = std::stoi(token.substr(0, token.find('/'))) - 1;
        n[i] = std::stoi(token.substr(token.find('/') + 2)) - 1;
      }
      triangles.push_back((Triangle){vertices[v[0]], vertices[v[1]], vertices[v[2]]});
      triangle_normals.push_back((TriangleNormal){normals[n[0]], normals[n[1]], normals[n[2]]});
      triangle_emissivities.push_back((TriangleEmissivity){make_float3(0.f)});
    }
  }

  Scene scene;
  scene.n_triangles  = triangles.size();
  scene.triangles    = triangles.data();
  scene.normals      = triangle_normals.data();
  scene.emissivities = triangle_emissivities.data();

  DEBUG_PRINT("Loaded scene with %d triangles\n", scene.n_triangles);

  return scene;
}
