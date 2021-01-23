#pragma once

#include "cu_math.h"

struct Ray {
  float3 origin;
  float3 direction;
};

struct Triangle {
  float3 v0;
  float3 v1;
  float3 v2;
};

struct TriangleNormal {
  float3 n0;
  float3 n1;
  float3 n2;
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

HD TriangleHitRecord hit_mt(const Ray& ray, const Triangle& tri);
IHD TriangleHitRecord hit(const Ray& ray, const Triangle& tri) {
  return hit_mt(ray, tri);
}

HD HitRecord hit(const Ray& ray, const AABB& bbox);
