#include "triangle.cuh"

#define FLT_MIN 1.175494351e-38F

bool Triangle::intersects(Ray r) const {
  float3 edge0 = v1 - v0;
  float3 edge1 = v2 - v0;

  float3 h = cross(r.d, edge1);

  float det = dot(edge0, h);

  if (fabsf(det) < FLT_MIN)
    return false;

  float detInv = 1.0f / det;
  float3 s     = r.o - v0;
  float u      = detInv * dot(s, h);

  if (u < 0.0f || u > 1.0f)
    return false;

  float3 q = cross(s, edge0);
  float v  = detInv * dot(r.d, q);

  if (v < 0.0f || u + v > 1.0f)
    return false;

  float time = detInv * dot(edge1, q);

  if (time < 0.01f)
    return false;

  return true;
}
