#pragma once

#include "alloc.cuh"
#include "triangle.cuh"

struct TriangleArrayIntersection {
  Triangle tri;
  Point3 point;
  Vec3 normal;
  float3 uvw;
  float time;
  int idx;
  bool hit;
};

struct TriangleArray {
  Vector<Triangle> tris;
  Vector<TriangleNormals> tri_normals;

  __host__ __device__ TriangleArrayIntersection intersects(Ray r) const;
  __host__ __device__ TriangleArrayIntersection intersects(Ray r, int idx) const;
};
