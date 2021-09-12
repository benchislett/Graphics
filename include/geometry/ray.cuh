#pragma once

#include "point.cuh"
#include "vector.cuh"

struct Ray {
  Point3 o;
  Vec3 d;

  __host__ __device__ Ray(Point3 o_, Vec3 d_) : o(o_), d(normalized(d_)) {}
  __host__ __device__ Point3 at(float t) const;
};
