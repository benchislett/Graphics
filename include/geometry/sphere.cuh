#pragma once

#include "float3.cuh"
#include "ray.cuh"


struct SphereIntersection {
  float3 normal;
  float3 point;
  float time;
  bool hit;
};

struct Sphere {
  float3 center;
  float radius;

  __host__ __device__ Sphere(float3 c, float r) : center(c), radius(r) {}

  __host__ __device__ SphereIntersection intersects(Ray r) const;
};
