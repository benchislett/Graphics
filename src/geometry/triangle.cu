#include "float.cuh"
#include "float3.cuh"
#include "triangle.cuh"

#include <cassert>

#define FLT_MIN 1.175494351e-38F

__host__ __device__ TriangleIntersection Triangle::intersects(Ray r) const {
  TriangleIntersection no_hit = {};

  Vec3 edge0 = v1 - v0;
  Vec3 edge1 = v2 - v0;

  float3 h = cross(r.d, edge1);

  float det = dot(edge0, h);

  if (fabsf(det) < FLT_MIN)
    return no_hit;

  float detInv = 1.0 / det;
  float3 s     = r.o - v0;
  float u      = detInv * dot(s, h);

  if (u < 0.0 || u > 1.0)
    return no_hit;

  float3 q = cross(s, edge0);
  float v  = detInv * dot(r.d, q);

  if (v < 0.0 || u + v > 1.0)
    return no_hit;

  float time = detInv * dot(edge1, q);

  if (time < 0.01)
    return no_hit;

  Point3 point = r.at(time);
  float3 uvw{u, v, 1.0f - u - v};

  return {point, uvw, time, true};
}

__host__ __device__ Vec3 TriangleNormals::at(float3 uvw) const {
  return normalized(uvw.z * n0 + uvw.x * n1 + uvw.y * n2);
}

__host__ __device__ Vec3 TriangleNormals::at(float3 uvw, Ray r) const {
  Vec3 normal    = at(uvw);
  int front_face = -sign(dot(r.d, normal));
  return normal * front_face;
}
