#pragma once

#include "primitive.cuh"
#include "bvh.cuh"

struct Scene {
  int primitiveCount;
  Primitive *primitives;

  int lightCount;
  int *lights;

  BVH bvh;
};
