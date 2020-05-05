#pragma once

#include "math.cuh"
#include "cuda.cuh"

struct Ray {
  Vec3 o;
  Vec3 d;

  Ray(const Vec3 &o, const Vec3 &d) : o(o), d(normalized(d)) {}

  __device__ Vec3 at(float t) const;
};
