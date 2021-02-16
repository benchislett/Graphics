#pragma once

#include "geometry.cuh"

#include <string>

struct Scene {
  int n_triangles;

  Triangle* triangles;
  TriangleNormal* normals;
  TriangleEmissivity* emissivities;

  int n_lights;
  int* lights;
};

Scene from_obj(const std::string& filename);
