#include "primitive.cuh"

__host__ __device__ float3 Primitive::sample(float u, float v) const {
  float u_ = sqrtf(u);
  float v_ = u_ * (1.f - v);
  float w_ = u_ * v;
  u_ = 1.f - u_;

  return tri.a * u_ + tri.b * v_ + tri.c * w_;
}

__host__ __device__ float Primitive::area() const {
  return 0.5f * length(cross(tri.b - tri.a, tri.c - tri.a));
}
