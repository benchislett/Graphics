#pragma once

#include "math.cuh"
#include "cuda.cuh"

struct BeckmannDistribution {
  float alpha_x, alpha_y;

  // Always samples visible area
  Beckmann(float ax, float ay), alpha_x(ax), alpha_y(ay) {}
  Beckmann(float roughness);

  __device__ float lambda(const Vec3 &w) const;
  __device__ Vec3 sample_wh(const Vec3 &wo, float u, float v) const;
  __device__ float d(const Vec3 &wh) const;
  __device__ float g1(const Vec3 &w) const;
  __device__ float g(const Vec3 &wo, const Vec3 &wi) const;
  __device__ float pdf(const Vec3 &wo, const Vec3 &wh) const;
};
