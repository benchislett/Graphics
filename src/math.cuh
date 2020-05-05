#pragma once

#include "cuda.cuh"

#include <math.h>
#include <cfloat>
#include <cstdint>
#include <cstdio>

#define TWO_PI 6.28318530718f
#define PI 3.14159265359f
#define INV_PI 0.31830988618f
#define SQRT_INV_PI 0.56418958354f
#define PI_OVER_4 0.78539816339f
#define PI_OVER_2 1.57079632679f

__host__ __device__ float lerp(float t, float a, float b);

struct Vec3 {
  float e[3];
  
  __host__ __device__ Vec3();
  __host__ __device__ Vec3(float w);
  __host__ __device__ Vec3(float x, float y, float z);

  __host__ __device__ Vec3(const Vec3 &v);
  __host__ __device__ Vec3 &operator=(const Vec3 &v);

  __host__ __device__ Vec3 operator+(const Vec3 &v) const;
  __host__ __device__ Vec3 &operator+=(const Vec3 &v);

  __host__ __device__ Vec3 operator-(const Vec3 &v) const;
  __host__ __device__ Vec3 &operator-=(const Vec3 &v);

  __host__ __device__ bool operator==(const Vec3 &v) const;
  __host__ __device__ bool operator!=(const Vec3 &v) const;

  __host__ __device__ Vec3 operator*(const float f) const;
  __host__ __device__ Vec3 &operator*=(const float f);

  __host__ __device__ Vec3 operator*(const Vec3 &v) const;
  __host__ __device__ Vec3 &operator*=(const Vec3 &v);

  __host__ __device__ Vec3 operator/(const float f) const;
  __host__ __device__ Vec3 &operator/=(const float f);

  __host__ __device__ Vec3 operator/(const Vec3 &v) const;
};

__host__ __device__ Vec3 operator*(float x, const Vec3 &v);

__host__ __device__ bool is_zero(const Vec3 &v);
__host__ __device__ bool has_nans(const Vec3 &v);

__host__ __device__ float length_sq(const Vec3 &v);
__host__ __device__ float length(const Vec3 &v);

__host__ __device__ float dot(const Vec3 &v1, const Vec3 &v2);
__host__ __device__ float dot_abs(const Vec3 &v1, const Vec3 &v2);

__host__ __device__ Vec3 cross(const Vec3 &v1, const Vec3 &v2);

__host__ __device__ Vec3 abs(const Vec3 &v);

__host__ __device__ Vec3 lerp(const float t, const Vec3 &v1, const Vec3 &v2);

__host__ __device__ Vec3 max(const Vec3 &v1, const Vec3 &v2);

__host__ __device__ Vec3 min(const Vec3 &v1, const Vec3 &v2);

__host__ __device__ Vec3 normalized(const Vec3 &v);
