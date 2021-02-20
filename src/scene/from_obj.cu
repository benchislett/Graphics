#include "scene.cuh"

#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

HostScene from_obj(const std::string& filename) {
  HostScene scene;
  std::vector<float3> vertices;
  std::vector<float3> normals;

  scene.diffuse_materials.resize(2);
  scene.diffuse_materials[0] = (DiffuseBRDF){make_float3(1.f, 1.f, 1.f)};

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
      scene.triangles.push_back((Triangle){vertices[v[0]], vertices[v[1]], vertices[v[2]]});
      scene.normals.push_back((TriangleNormal){normals[n[0]], normals[n[1]], normals[n[2]]});
      scene.emissivities.push_back(current_emit);
      scene.material_ids.push_back(scene.diffuse_materials.size - 1);
      if (length(current_emit.intensity) > 0.f) {
        scene.lights.push_back(scene.triangles.size - 1);
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

  scene.n_triangles = scene.triangles.size;
  scene.n_lights    = scene.lights.size;

  return scene;
}
