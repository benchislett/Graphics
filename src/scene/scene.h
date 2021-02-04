#pragma once

#include "geometry.h"

#include <string>

struct Scene {
  int n_triangles;

  const Triangle* triangles;
  const TriangleNormal* normals;
  const TriangleEmissivity* emissivities;
};

Scene from_obj(const std::string& filename);
