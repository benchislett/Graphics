#pragma once
#include "../math/float3.cuh"
#include "primitive.cuh"

struct AABB : Primitive {
  float3 lo;
  float3 hi;
  AABB(float3 l, float3 h) : lo(l), hi(h) {}

  bool intersects(Ray r) const;
};
