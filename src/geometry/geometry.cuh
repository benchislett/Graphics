#pragma once

#include "cu_math.cuh"

struct Ray {
  float3 origin;
  float3 direction;
};

struct Triangle {
  float3 v0;
  float3 v1;
  float3 v2;
};

HD float3 interp(const Triangle tri, float u, float v);

struct TriangleNormal {
  float3 n0;
  float3 n1;
  float3 n2;
};

HD float3 interp(const TriangleNormal tri, float u, float v);

struct TriangleEmissivity {
  float3 intensity;
};

struct AABB {
  float3 lo;
  float3 hi;
};

struct HitRecord {
  float time;
  bool hit;
};

struct TriangleHitRecord {
  float time;
  float u, v;
  bool hit;
};

HD TriangleHitRecord hit_mt(const Ray ray, const Triangle tri);
IHD TriangleHitRecord hit(const Ray ray, const Triangle tri) {
  return hit_mt(ray, tri);
}

HD TriangleHitRecord first_hit(const Ray ray, const Triangle* triangles, int n_triangles, int* which);

HD HitRecord hit(const Ray ray, const AABB bbox);
