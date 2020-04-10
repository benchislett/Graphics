#pragma once

#include "math.cuh"
#include "primitive.cuh"
#include "camera.cuh"
#include "bvh.cuh"

#include <cuda_runtime.h>

struct Scene {
  Camera cam;
  BVH b;
  Vec3 background;
  int spp;
};

struct Image {
  int width;
  int height;
  Vec3 *film; // row-major
};

void Render(const Scene &scene, Image &im);
