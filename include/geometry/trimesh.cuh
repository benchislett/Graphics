#pragma once

#include "triangle.cuh"

struct TriMeshIntersection {
  Triangle tri;
  float3 point;
  float3 uvw;
  float time;
  bool hit;
};


struct TriMesh {
  Triangle* tris;
  int n;

  __host__ __device__ TriMesh(Triangle* ts, int n) : tris(ts), n(n) {}

  __host__ __device__ TriMeshIntersection intersects(Ray r) const;
};
