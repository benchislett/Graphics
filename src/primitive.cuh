#pragma once

#include "helper_math.cuh"

struct AABB {
  float3 lower;
  float3 upper;
};

struct Tri {
  float3 a;
  float3 b;
  float3 c;

  float3 normalA;
  float3 normalB;
  float3 normalC;
};

struct Primitive {
  Tri tri;
  AABB bound;
  float emittance;

  __host__ __device__ Primitive() {}
  __host__ __device__ Primitive(const Tri &t, const AABB &b, float e) : tri(t), bound(b), emittance(e) {}

  __host__ __device__ float3 sample(float u, float v) const;
  __host__ __device__ float area() const;
};

