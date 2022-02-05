#include "obj.cuh"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <iostream>

OBJScene load_obj(const std::string& filename) {
  OBJScene scene{};

  tinyobj::ObjReaderConfig reader_config;
  tinyobj::ObjReader reader;

  if (!reader.ParseFromFile(filename, reader_config)) {
    if (!reader.Error().empty()) {
      std::cerr << "TinyObjReader: " << reader.Error();
    }
    exit(1);
  }


  if (!reader.Warning().empty()) {
    std::cout << "TinyObjReader: " << reader.Warning();
  }

  auto& attrib    = reader.GetAttrib();
  auto& shapes    = reader.GetShapes();
  auto& materials = reader.GetMaterials();

  for (int i = 0; i < shapes.size(); i++) {
    int index_offset = 0;
    for (int fv : shapes[i].mesh.num_face_vertices) {
      assert(fv == 3);

      Point3 vertices[3];
      Vec3 normals[3];
      bool has_normals = false;

      for (int v = 0; v < fv; v++) {
        // access to vertex
        auto idx = shapes[i].mesh.indices[index_offset + v];

        vertices[v].x = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
        vertices[v].y = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
        vertices[v].z = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

        // Check if `normal_index` is zero or positive. negative = no normal data
        if (idx.normal_index >= 0) {
          has_normals  = true;
          normals[v].x = attrib.normals[3 * size_t(idx.normal_index) + 0];
          normals[v].y = attrib.normals[3 * size_t(idx.normal_index) + 1];
          normals[v].z = attrib.normals[3 * size_t(idx.normal_index) + 2];
        }
      }
      index_offset += fv;

      Triangle tri(vertices[0], vertices[1], vertices[2]);
      scene.primitives.tris.push_back(tri);
      if (has_normals) {
        scene.primitives.tri_normals.push_back(TriangleNormals(normals[0], normals[1], normals[2]));
      } else {
        scene.primitives.tri_normals.push_back(TriangleNormals(tri));
      }
    }
  }

  // std::string line;
  // std::ifstream file(filename);

  // if (file.is_open()) {
  //   while (getline(file, line)) {
  //     std::istringstream iss(line);
  //     rtrim(line);
  //     std::string token;

  //     iss >> token;

  //     if (token == "v") {
  //       Point3 vertex;
  //       sscanf(line.c_str(), "v  %f %f %f", &vertex.x, &vertex.y, &vertex.z);
  //       scene.vertices.push_back(vertex);
  //     } else if (token == "vn") {
  //       Point3 normal;
  //       sscanf(line.c_str(), "vn %f %f %f", &normal.x, &normal.y, &normal.z);
  //       scene.normals.push_back(normal);
  //     } else if (token == "f") {
  //       int vertices[3];
  //       int textures[3];
  //       int normals[3];

  //       int ret = sscanf(line.c_str(), "f %d//%d %d//%d %d//%d", &vertices[0], &normals[0], &vertices[1],
  //       &normals[1],
  //                        &vertices[2], &normals[2]);

  //       if (ret < 6) {
  //         ret = sscanf(line.c_str(), "f %d/%d/%d %d/%d/%d %d/%d/%d", &vertices[0], &textures[0], &normals[0],
  //                      &vertices[1], &textures[1], &normals[1], &vertices[2], &textures[2], &normals[2]);
  //       }

  //       if (ret >= 6) {
  //         Triangle tri(scene.vertices[vertices[0] - 1], scene.vertices[vertices[1] - 1],
  //                      scene.vertices[vertices[2] - 1]);
  //         scene.primitives.tris.push_back(tri);
  //         scene.primitives.tri_normals.push_back(TriangleNormals(
  //             scene.normals[normals[0] - 1], scene.normals[normals[1] - 1], scene.normals[normals[2] - 1]));
  //       }
  //     }
  //   }
  //   file.close();
  // }

  std::cout << "Loaded scene with " << scene.primitives.tris.size << " triangles\n";

  return scene;
}