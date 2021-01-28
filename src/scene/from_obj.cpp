#include "scene.h"

#include <fstream>
#include <sstream>
#include <utility>
#include <vector>

Scene from_obj(const std::string& filename) {
  std::vector<float3> vertices;
  std::vector<float3> normals;
  std::vector<Triangle> triangles;
  std::vector<TriangleNormal> triangle_normals;
  std::vector<TriangleEmissivity> triangle_emissivities;

  std::ifstream file;
  file.open(filename, std::ios::in);

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

  // Move the vectors so they don't expire
  std::move(triangles);
  std::move(triangle_normals);
  std::move(triangle_emissivities);
  return scene;
}
