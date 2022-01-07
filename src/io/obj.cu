#include "obj.cuh"

#include <fstream>
#include <iostream>
#include <sstream>

static inline void rtrim(std::string& s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
}

OBJScene load_obj(const std::string& filename) {
  OBJScene scene{};

  std::string line;
  std::ifstream file(filename);

  if (file.is_open()) {
    while (getline(file, line)) {
      std::istringstream iss(line);
      rtrim(line);
      std::string token;

      iss >> token;

      if (token == "v") {
        Point3 vertex;
        sscanf(line.c_str(), "v  %f %f %f", &vertex.x, &vertex.y, &vertex.z);
        scene.vertices.push_back(vertex);
      } else if (token == "vn") {
        Point3 normal;
        sscanf(line.c_str(), "vn %f %f %f", &normal.x, &normal.y, &normal.z);
        scene.normals.push_back(normal);
      } else if (token == "f") {
        int vertices[3];
        int textures[3];
        int normals[3];

        int ret = sscanf(line.c_str(), "f %d//%d %d//%d %d//%d", &vertices[0], &normals[0], &vertices[1], &normals[1],
                         &vertices[2], &normals[2]);

        if (ret < 6) {
          ret = sscanf(line.c_str(), "f %d/%d/%d %d/%d/%d %d/%d/%d", &vertices[0], &textures[0], &normals[0],
                       &vertices[1], &textures[1], &normals[1], &vertices[2], &textures[2], &normals[2]);
        }

        if (ret >= 6) {
          scene.primitives.tris.push_back(Triangle(scene.vertices[vertices[0] - 1], scene.vertices[vertices[1] - 1],
                                                   scene.vertices[vertices[2] - 1]));
          scene.primitives.tri_normals.push_back(TriangleNormals(
              scene.normals[normals[0] - 1], scene.normals[normals[1] - 1], scene.normals[normals[2] - 1]));
        }
      }
    }
    file.close();
  }

  return scene;
}