#pragma once

#include "geometry.h"

#include <string>

struct Scene {
  int n_triangles;

  const Triangle* triangles;
  const TriangleNormal* normals;
  const TriangleEmissivity* emissivities;

  int n_lights;
  const int* lights;
};

Scene from_obj(const std::string& filename);
