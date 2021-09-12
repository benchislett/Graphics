#pragma once

#include "point.cuh"
#include "ray.cuh"
#include "vector.cuh"

struct SphereIntersection {
  Vec3 normal;
  Point3 point;
  float time;
  bool hit;
};

struct Sphere {
  Point3 center;
  float radius;

  __host__ __device__ Sphere(Point3 c, float r) : center(c), radius(r) {}

  __host__ __device__ SphereIntersection intersects(Ray r) const;
};
