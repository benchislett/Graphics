#pragma once

#include "../math/float3.cuh"
#include "point.cuh"
#include "ray.cuh"
#include "vector.cuh"

struct TriangleIntersection {
  Point3 point;
  float3 uvw;
  float time;
  bool hit;
};

struct Triangle {
  Point3 v0, v1, v2;

  __host__ __device__ Triangle() {}
  __host__ __device__ Triangle(Point3 a, Point3 b, Point3 c) : v0(a), v1(b), v2(c) {}

  __host__ __device__ TriangleIntersection intersects(Ray r) const;
};

struct TriangleNormals {
  Vec3 n0, n1, n2;

  __host__ __device__ TriangleNormals(Vec3 a, Vec3 b, Vec3 c)
      : n0(normalized(a)), n1(normalized(b)), n2(normalized(c)) {}
  __host__ __device__ TriangleNormals(Vec3 a) : TriangleNormals(a, a, a) {}
  __host__ __device__ TriangleNormals(Triangle t) : TriangleNormals(cross(t.v2 - t.v0, t.v1 - t.v0)) {}

  __host__ __device__ Vec3 at(float3 uvw) const;
  __host__ __device__ Vec3 at(float3 uvw, Ray r) const; // Ensures normals are always facing towards the ray
};
