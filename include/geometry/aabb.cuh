#pragma once
#include "../math/float3.cuh"
#include "ray.cuh"

struct AABBIntersection {
  float time;
  bool hit;
};

struct AABB {
  float3 lo;
  float3 hi;
  __host__ __device__ AABB(float3 l, float3 h) : lo(l), hi(h) {}

  __host__ __device__ AABBIntersection intersects(Ray r) const;
};
