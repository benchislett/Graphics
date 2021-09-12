#pragma once

#include "float.cuh"

#include <cuda_runtime.h>

__host__ __device__ float dot(float3 a, float3 b);

__host__ __device__ float length(float3 v);

__host__ __device__ float fminf(float3 a);

__host__ __device__ float fmaxf(float3 a);

__host__ __device__ void operator+=(float3& a, float3 b);

__host__ __device__ void operator-=(float3& a, float3 b);

__host__ __device__ void operator*=(float3& a, float3 b);

__host__ __device__ void operator*=(float3& a, float b);

__host__ __device__ void operator/=(float3& a, float3 b);

__host__ __device__ void operator/=(float3& a, float b);

__host__ __device__ float3 normalized(float3 v);

__host__ __device__ float3 operator-(float3 a);

__host__ __device__ float3 operator+(float3 a, float3 b);

__host__ __device__ float3 operator+(float3 a, float b);

__host__ __device__ float3 operator+(float b, float3 a);

__host__ __device__ float3 operator-(float3 a, float3 b);

__host__ __device__ float3 operator-(float3 a, float b);

__host__ __device__ float3 operator-(float b, float3 a);

__host__ __device__ float3 operator*(float3 a, float3 b);

__host__ __device__ float3 operator*(float3 a, float b);

__host__ __device__ float3 operator*(float b, float3 a);

__host__ __device__ float3 operator/(float3 a, float3 b);

__host__ __device__ float3 operator/(float3 a, float b);

__host__ __device__ float3 operator/(float b, float3 a);

__host__ __device__ float3 fminf(float3 a, float3 b);

__host__ __device__ float3 fmaxf(float3 a, float3 b);

__host__ __device__ float3 lerp(float3 a, float3 b, float t);

__host__ __device__ float3 clamp(float3 v, float a, float b);

__host__ __device__ float3 clamp(float3 v, float3 a, float3 b);

__host__ __device__ float3 fabsf(float3 v);

__host__ __device__ float3 cross(float3 a, float3 b);
