#include "float.cuh"
#include "float3.cuh"

__host__ __device__ float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ float length(float3 v) {
  return sqrtf(dot(v, v));
}

__host__ __device__ float fminf(float3 a) {
  return fminf(fminf(a.x, a.y), a.z);
}

__host__ __device__ float fmaxf(float3 a) {
  return fmaxf(fmaxf(a.x, a.y), a.z);
}

__host__ __device__ void operator+=(float3& a, float3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

__host__ __device__ void operator-=(float3& a, float3 b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}

__host__ __device__ void operator*=(float3& a, float3 b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
}

__host__ __device__ void operator*=(float3& a, float b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
}

__host__ __device__ void operator/=(float3& a, float3 b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
}

__host__ __device__ void operator/=(float3& a, float b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
}

__host__ __device__ float3 normalized(float3 v) {
  float invLen = rsqrtf(dot(v, v));
  return v * invLen;
}

__host__ __device__ float3 operator-(float3 a) {
  return {-a.x, -a.y, -a.z};
}

__host__ __device__ float3 operator+(float3 a, float3 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ float3 operator+(float3 a, float b) {
  return {a.x + b, a.y + b, a.z + b};
}

__host__ __device__ float3 operator+(float b, float3 a) {
  return {a.x + b, a.y + b, a.z + b};
}

__host__ __device__ float3 operator-(float3 a, float3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ float3 operator-(float3 a, float b) {
  return {a.x - b, a.y - b, a.z - b};
}

__host__ __device__ float3 operator-(float b, float3 a) {
  return {b - a.x, b - a.y, b - a.z};
}

__host__ __device__ float3 operator*(float3 a, float3 b) {
  return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__host__ __device__ float3 operator*(float3 a, float b) {
  return {a.x * b, a.y * b, a.z * b};
}

__host__ __device__ float3 operator*(float b, float3 a) {
  return {a.x * b, a.y * b, a.z * b};
}

__host__ __device__ float3 operator/(float3 a, float3 b) {
  return {a.x / b.x, a.y / b.y, a.z / b.z};
}

__host__ __device__ float3 operator/(float3 a, float b) {
  return {a.x / b, a.y / b, a.z / b};
}

__host__ __device__ float3 operator/(float b, float3 a) {
  return {b / a.x, b / a.y, b / a.z};
}


__host__ __device__ float3 fminf(float3 a, float3 b) {
  return {fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)};
}

__host__ __device__ float3 fmaxf(float3 a, float3 b) {
  return {fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)};
}

__host__ __device__ float3 lerp(float3 a, float3 b, float t) {
  return a + t * (b - a);
}

__host__ __device__ float3 clamp(float3 v, float a, float b) {
  return {fclamp(v.x, a, b), fclamp(v.y, a, b), fclamp(v.z, a, b)};
}

__host__ __device__ float3 clamp(float3 v, float3 a, float3 b) {
  return {fclamp(v.x, a.x, b.x), fclamp(v.y, a.y, b.y), fclamp(v.z, a.z, b.z)};
}

__host__ __device__ float3 fabsf(float3 v) {
  return {fabsf(v.x), fabsf(v.y), fabsf(v.z)};
}

__host__ __device__ float3 cross(float3 a, float3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
