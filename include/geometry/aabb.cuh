#pragma once

#include "point.cuh"
#include "ray.cuh"
#include "triangle.cuh"

const float inf  = __FLT_MAX__;
const float minf = -inf;

struct AABBIntersection {
  float time;
  bool hit;
};

struct AABB {
  Point3 lo;
  Point3 hi;

  __host__ __device__ AABB() : lo{inf, inf, inf}, hi{minf, minf, minf} {}
  __host__ __device__ AABB(Point3 l, Point3 h) : lo(l), hi(h) {}
  __host__ __device__ AABB(Triangle tri)
      : lo(fminf(fminf(tri.v0, tri.v1), tri.v2)), hi(fmaxf(fmaxf(tri.v0, tri.v1), tri.v2)) {}

  __host__ __device__ AABBIntersection intersects(Ray r) const;
  __host__ __device__ AABB plus(AABB other) const;

  __host__ __device__ Point3 centroid() const;
  __host__ __device__ Vec3 length() const;

  __host__ __device__ float surface_area() const;
};
