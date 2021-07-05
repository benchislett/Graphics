#pragma once
#include "../math/float3.cuh"
#include "ray.cuh"

struct TriangleIntersection {
  float3 point;
  float3 uvw;
  float time;
  bool hit;
};

struct Triangle {
  float3 v0, v1, v2;

  Triangle(float3 a, float3 b, float3 c) : v0(a), v1(b), v2(c) {}

  TriangleIntersection intersects(Ray r) const;
};
