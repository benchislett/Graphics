#pragma once

#include "cu_misc.cuh"

#include <cfloat>

// Host-only code
#ifndef __CUDACC__
#include <cmath>

IHD float rsqrtf(float s) {
  return 1.f / sqrtf(s);
}

#endif

// Float -> Float //

IHD float clamp(float v, float a, float b) {
  return v < a ? a : (v > b ? b : v);
}

// Float -> Float3

IHD float3 make_float3(float r) {
  return make_float3(r, r, r);
}

// Float3 -> Float //

IHD float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

IHD float length(float3 v) {
  return sqrtf(dot(v, v));
}

IHD float fminf(float3 a) {
  return fminf(fminf(a.x, a.y), a.z);
}

IHD float fmaxf(float3 a) {
  return fmaxf(fmaxf(a.x, a.y), a.z);
}

// Float3 -> Void //

IHD void operator+=(float3& a, float3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

IHD void operator-=(float3& a, float3 b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}

IHD void operator*=(float3& a, float3 b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
}

IHD void operator*=(float3& a, float b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
}

IHD void operator/=(float3& a, float3 b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
}

IHD void operator/=(float3& a, float b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
}

IHD void normalize(float3& v) {
  float invLen = rsqrtf(dot(v, v));
  v *= invLen;
}


// Float3 -> Float3 //

IHD float3 operator-(float3 a) {
  return make_float3(-a.x, -a.y, -a.z);
}

IHD float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

IHD float3 operator+(float3 a, float b) {
  return make_float3(a.x + b, a.y + b, a.z + b);
}

IHD float3 operator+(float b, float3 a) {
  return make_float3(a.x + b, a.y + b, a.z + b);
}

IHD float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

IHD float3 operator-(float3 a, float b) {
  return make_float3(a.x - b, a.y - b, a.z - b);
}

IHD float3 operator-(float b, float3 a) {
  return make_float3(b - a.x, b - a.y, b - a.z);
}

IHD float3 operator*(float3 a, float3 b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

IHD float3 operator*(float3 a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}

IHD float3 operator*(float b, float3 a) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}

IHD float3 operator/(float3 a, float3 b) {
  return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

IHD float3 operator/(float3 a, float b) {
  return make_float3(a.x / b, a.y / b, a.z / b);
}

IHD float3 operator/(float b, float3 a) {
  return make_float3(b / a.x, b / a.y, b / a.z);
}

IHD float3 normalized(float3 a) {
  normalize(a);
  return a;
}

IHD float3 fminf(float3 a, float3 b) {
  return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

IHD float3 fmaxf(float3 a, float3 b) {
  return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

IHD float3 lerp(float3 a, float3 b, float t) {
  return a + t * (b - a);
}

IHD float3 clamp(float3 v, float a, float b) {
  return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

IHD float3 clamp(float3 v, float3 a, float3 b) {
  return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

IHD float3 fabsf(float3 v) {
  return make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
}

IHD float3 reflect(float3 incident, float3 normal) {
  return incident - normal * (2.0f * dot(normal, incident));
}

IHD float3 cross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
