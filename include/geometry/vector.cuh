#pragma once

#include "../math/float3.cuh"

struct Vec3 : float3 {
  __host__ __device__ constexpr Vec3() : float3() {}
  __host__ __device__ constexpr Vec3(float x, float y, float z) : float3{x, y, z} {}
  __host__ __device__ constexpr Vec3(float3 x) : float3(x) {}
};
