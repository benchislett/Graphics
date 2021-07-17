#pragma once
#include "../math/float3.cuh"
#include "ray.cuh"

struct TriangleIntersection {
  float3 point;
  float3 uvw;
  float time;
  bool hit;
};

struct Triangle {
  float3 v0, v1, v2;

  __host__ __device__ Triangle(float3 a, float3 b, float3 c) : v0(a), v1(b), v2(c) {}

  __host__ __device__ TriangleIntersection intersects(Ray r) const;
};

struct TriangleNormals {
  float3 n0, n1, n2;

  __host__ __device__ TriangleNormals(float3 a, float3 b, float3 c) : n0(a), n1(b), n2(c) {}
  __host__ __device__ TriangleNormals(float3 a) : TriangleNormals(a, a, a) {}
  __host__ __device__ TriangleNormals(Triangle t) : TriangleNormals(cross(t.v2 - t.v0, t.v1 - t.v0)) {}

  __host__ __device__ float3 at(float3 uvw) const;
  __host__ __device__ float3 at(float3 uvw, Ray r) const; // Ensures normals are always facing towards the ray
};
