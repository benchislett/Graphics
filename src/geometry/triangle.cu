#include "geometry.cuh"

HD float3 interp(const Triangle tri, float u, float v) {
  float w = 1.f - u - v;
  return (w * tri.v0) + (u * tri.v1) + (v * tri.v2);
}

HD float3 interp(const TriangleNormal tri, float u, float v) {
  float w = 1.f - u - v;
  return (w * tri.n0) + (u * tri.n1) + (v * tri.n2);
}