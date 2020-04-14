#pragma once

#include "math.cuh"
#include "scene.cuh"

struct RenderParams {
  int spp;
};

struct Image {
  int width;
  int height;
  Vec3 *film; // row-major
};

void Render(const RenderParams &params, const Scene &scene, Image &im);
