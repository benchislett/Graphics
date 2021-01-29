#pragma once

#include "geometry.h"

#include <string>

struct Image {
  int x;
  int y;
  float3* data; // [0,1]
};

void to_ppm(const Image image, const std::string& filename);

struct Scene {
  int n_triangles;

  const Triangle* triangles;
  const TriangleNormal* normals;
  const TriangleEmissivity* emissivities;
};

Scene from_obj(const std::string& filename);
