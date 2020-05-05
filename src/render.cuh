#pragma once

#include "math.cuh"
#include "vector.cuh"
#include "cuda.cuh"
#include "scene.cuh"

struct RenderParams {
  int spp;
};

struct Image {
  const int width;
  const int height;
  Vector<Vec3> film; // row-major
  
  Image(int width, int height) : width(width), height(height), film { width * height } {}

  void to_host() { film.to_host(); }
  void to_device() { film.to_device(); }
};

__host__ void render(const RenderParams &params, Scene &scene, Image &im);
