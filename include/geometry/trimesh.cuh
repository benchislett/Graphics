#pragma once

#include "alloc.cuh"
#include "triangle.cuh"

struct TriMeshIntersection {
  Triangle tri;
  Point3 point;
  float3 uvw;
  float time;
  bool hit;
};

struct TriMesh {
  Vector<Triangle> tris;

  __host__ __device__ TriMeshIntersection intersects(Ray r) const;
};
