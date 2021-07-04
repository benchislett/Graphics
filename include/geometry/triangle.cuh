#pragma once
#include "../math/float3.cuh"
#include "primitive.cuh"

struct Triangle : Primitive {
  float3 v0, v1, v2;

  Triangle(float3 a, float3 b, float3 c) : v0(a), v1(b), v2(c) {}

  bool intersects(Ray r) const;
};
