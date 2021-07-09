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

  Triangle(float3 a, float3 b, float3 c) : v0(a), v1(b), v2(c) {}

  TriangleIntersection intersects(Ray r) const;
};

struct TriangleNormals {
  float3 n0, n1, n2;

  TriangleNormals(float3 a) : n0(a), n1(a), n2(a) {}
  TriangleNormals(float3 a, float3 b, float3 c) : n0(a), n1(b), n2(c) {}

  float3 at(float3 uvw) const;
  float3 at(float3 uvw, Ray r) const; // Ensures normals are always facing towards the ray
};
