#pragma once

#include "alloc.cuh"
#include "triangle.cuh"

struct TriangleArrayIntersection {
  Triangle tri;
  Point3 point;
  float3 uvw;
  float time;
  int idx;
  bool hit;
};

struct TriangleArray : Vector<Triangle> {
  using Vector<Triangle>::Vector;
  __host__ __device__ TriangleArrayIntersection intersects(Ray r) const;
};
