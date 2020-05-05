#pragma once

#include "math.cuh"
#include "scene.cuh"

struct RenderParams {
  int spp;
};

struct Image {
  const int width;
  const int height;
  Vector<Vec3> film; // row-major
  
  Image(int width, int height) : width(width), height(height), film { width * height } {}
};

void Render(const RenderParams &params, Scene &scene, Image &im);
