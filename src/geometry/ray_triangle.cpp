#include "geometry.h"

constexpr float hit_time_cutoff = 0.01f;

IHD TriangleHitRecord hit_mt(const Ray& ray, const Triangle& tri) {
  TriangleHitRecord record;
  record.hit = false;

  float3 edge0 = tri.v1 - tri.v0;
  float3 edge1 = tri.v2 - tri.v1;

  float3 h          = cross(ray.direction, edge1);
  float determinant = dot(edge0, h);

  if (fabsf(determinant) < FLT_MIN) {
    return record;
  }

  float determinantInv = 1.0f / determinant;
  float3 s             = ray.origin - tri.v0;
  float u              = determinantInv * dot(s, h);

  if (u < 0.f || u > 1.f) {
    return record;
  }

  float3 q = cross(s, edge0);
  float v  = determinantInv * dot(ray.direction, q);

  if (v < 0.f || u + v > 1.f) {
    return record;
  }

  float time = determinantInv * dot(edge1, q);

  if (time < hit_time_cutoff) {
    return record;
  }

  record.hit  = true;
  record.time = time;
  record.u    = u;
  record.v    = v;

  return record;
}

