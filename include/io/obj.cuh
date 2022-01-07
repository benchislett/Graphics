#pragma once

#include "alloc.cuh"
#include "point.cuh"
#include "tri_array.cuh"
#include "triangle.cuh"

#include <string>

struct OBJScene {
  Vector<Point3> vertices;
  Vector<Point3> normals;

  TriangleArray tris;
  Vector<TriangleNormals> tri_normals;
};

OBJScene load_obj(const std::string& filename);