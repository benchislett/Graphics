#pragma once

#include "../math/float3.cuh"

struct Ray {
  float3 o;
  float3 d;

  __host__ __device__ Ray(float3 o_, float3 d_) : o(o_), d(normalized(d_)) {}
  __host__ __device__ float3 at(float t) const;
};
