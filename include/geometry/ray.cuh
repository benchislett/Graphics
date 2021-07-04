#pragma once

#include "../math/float3.cuh"

struct Ray {
  float3 o;
  float3 d;

  Ray(float3 o_, float3 d_) : o(o_), d(d_) {}
};
