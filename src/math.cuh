#pragma once

#include <math.h>
#include <cfloat>
#include <cstdint>

float lerp(float t, float a, float b);

struct Vec3 {
  float e[3];
  
  Vec3();
  Vec3(float w);
  Vec3(float x, float y, float z);

  Vec3(const Vec3 &v);
  Vec3 &operator=(const Vec3 &v);

  Vec3 operator+(const Vec3 &v) const;
  Vec3 &operator+=(const Vec3 &v);

  Vec3 operator-(const Vec3 &v) const;
  Vec3 &operator-=(const Vec3 &v);

  bool operator==(const Vec3 &v) const;
  bool operator!=(const Vec3 &v) const;

  Vec3 operator*(const float f) const;
  Vec3 &operator*=(const float f);

  Vec3 operator*(const Vec3 &v) const;
  Vec3 &operator*=(const Vec3 &v);

  Vec3 operator/(const float f) const;
  Vec3 &operator/=(const float f);

  Vec3 operator/(const Vec3 &v) const;
};

Vec3 operator*(float x, const Vec3 &v);

bool is_zero(const Vec3 &v);
bool has_nans(const Vec3 &v);

float length_sq(const Vec3 &v);
float length(const Vec3 &v);

float dot(const Vec3 &v1, const Vec3 &v2);
float dot_abs(const Vec3 &v1, const Vec3 &v2);

Vec3 cross(const Vec3 &v1, const Vec3 &v2);

Vec3 abs(const Vec3 &v);

Vec3 lerp(const float t, const Vec3 &v1, const Vec3 &v2);

Vec3 max(const Vec3 &v1, const Vec3 &v2);

Vec3 min(const Vec3 &v1, const Vec3 &v2);

Vec3 normalized(const Vec3 &v);
