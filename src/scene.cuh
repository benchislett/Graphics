#pragma once

#include "math.cuh"
#include "camera.cuh"
#include "bvh.cuh"

struct Scene {
  Camera cam;
  BVH b;
  Vec3 background;
};
