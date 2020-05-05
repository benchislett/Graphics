#pragma once

#include "math.cuh"
#include "cuda.cuh"

__device__ float cos_theta(const Vec3 &w);
__device__ float cos2_theta(const Vec3 &w);
__device__ float abscos_theta(const Vec3 &w);
__device__ float sin_theta(const Vec3 &w);
__device__ float sin2_theta(const Vec3 &w);
__device__ float tan_theta(const Vec3 &w);
__device__ float tan2_theta(const Vec3 &w);

__device__ float cos_phi(const Vec3 &w);
__device__ float cos2_phi(const Vec3 &w);
__device__ float sin_phi(const Vec3 &w);
__device__ float sin2_phi(const Vec3 &w);

__device__ bool same_hemisphere(const Vec3 &w1, const Vec3 &w2);

__device__ Vec3 reflect(const Vec3 &w, const Vec3 &n);
__device__ bool refract(const Vec3 &w_in, const Vec3 &n, float eta, Vec3 *w_t);

__device__ Vec3 cosine_sample(float u, float v);

