#pragma once
#include "point.cuh"
#include "ray.cuh"

struct AABBIntersection {
  float time;
  bool hit;
};

struct AABB {
  Point3 lo;
  Point3 hi;
  __host__ __device__ AABB(Point3 l, Point3 h) : lo(l), hi(h) {}

  __host__ __device__ AABBIntersection intersects(Ray r) const;
};
