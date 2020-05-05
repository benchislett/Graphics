#include "math.cuh"

__host__ __device__ float lerp(float t, float a, float b) { return a + t * (b - a); }

__host__ __device__ Vec3::Vec3() { e[0] = 0.f; e[1] = 0.f; e[2] = 0.f; }
__host__ __device__ Vec3::Vec3(float w) { e[0] = w; e[1] = w; e[2] = w; }
__host__ __device__ Vec3::Vec3(float x, float y, float z) { e[0] = x; e[1] = y; e[2] = z; }

__host__ __device__ Vec3::Vec3(const Vec3 &v) { e[0] = v.e[0]; e[1] = v.e[1]; e[2] = v.e[2]; }
__host__ __device__ Vec3 &Vec3::operator=(const Vec3 &v) { e[0] = v.e[0]; e[1] = v.e[1]; e[2] = v.e[2]; return *this; }

__host__ __device__ Vec3 Vec3::operator+(const Vec3 &v) const { return Vec3(e[0] + v.e[0], e[1] + v.e[1], e[2] + v.e[2]); }
__host__ __device__ Vec3 &Vec3::operator+=(const Vec3 &v) { e[0] += v.e[0]; e[1] += v.e[1]; e[2] += v.e[2]; return *this; }

__host__ __device__ Vec3 Vec3::operator-(const Vec3 &v) const { return Vec3(e[0] - v.e[0], e[1] - v.e[1], e[2] - v.e[2]); }
__host__ __device__ Vec3 &Vec3::operator-=(const Vec3 &v) { e[0] -= v.e[0]; e[1] -= v.e[1]; e[2] -= v.e[2]; return *this; }

__host__ __device__ bool Vec3::operator==(const Vec3 &v) const { return e[0] == v.e[0] && e[1] == v.e[1] && e[2] == v.e[2]; }
__host__ __device__ bool Vec3::operator!=(const Vec3 &v) const { return e[0] != v.e[0] || e[1] != v.e[1] || e[2] != v.e[2]; }

__host__ __device__ Vec3 Vec3::operator*(const float f) const { return Vec3(e[0] * f, e[1] * f, e[2] * f); }
__host__ __device__ Vec3 &Vec3::operator*=(const float f) { e[0] *= f; e[1] *= f; e[2] *= f; return *this; }

__host__ __device__ Vec3 Vec3::operator*(const Vec3 &v) const { return Vec3(e[0] * v.e[0], e[1] * v.e[1], e[2] * v.e[2]); }
__host__ __device__ Vec3 &Vec3::operator*=(const Vec3 &v) { e[0] *= v.e[0]; e[1] *= v.e[1]; e[2] *= v.e[2]; return *this; }

__host__ __device__ Vec3 Vec3::operator/(const float f) const { return (*this) * (1.f / f); }
__host__ __device__ Vec3 &Vec3::operator/=(const float f) { return (*this) *= (1.f / f); }

__host__ __device__ Vec3 Vec3::operator/(const Vec3 &v) const { return Vec3(e[0] / v.e[0], e[1] / v.e[1], e[2] / v.e[2]); }

__host__ __device__ Vec3 operator*(float x, const Vec3 &v) {
  return v * x;
}

__host__ __device__ bool is_zero(const Vec3 &v) { return v.e[0] == 0.f && v.e[1] == 0.f && v.e[2] == 0.f; }
__host__ __device__ bool has_nans(const Vec3 &v) { return std::isnan(v.e[0]) || std::isnan(v.e[1]) || std::isnan(v.e[2]); }

__host__ __device__ float length_sq(const Vec3 &v) { return v.e[0] * v.e[0] + v.e[1] * v.e[1] + v.e[2] * v.e[2]; }
__host__ __device__ float length(const Vec3 &v) { return sqrtf(length_sq(v)); }

__host__ __device__ float dot(const Vec3 &v1, const Vec3 &v2) { return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2]; }
__host__ __device__ float dot_abs(const Vec3 &v1, const Vec3 &v2) { return std::fabs(dot(v1, v2)); }

__host__ __device__ Vec3 cross(const Vec3 &v1, const Vec3 &v2) {
  return Vec3(v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1],
              v1.e[2] * v2.e[0] - v1.e[0] * v2.e[2],
              v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]);
}

__host__ __device__ Vec3 abs(const Vec3 &v) { return Vec3(std::fabs(v.e[0]), std::fabs(v.e[1]), std::fabs(v.e[2])); }

__host__ __device__ Vec3 lerp(const float t, const Vec3 &v1, const Vec3 &v2) {
  return Vec3(lerp(t, v1.e[0], v2.e[0]), lerp(t, v1.e[1], v2.e[1]), lerp(t, v1.e[2], v2.e[2]));
}

__host__ __device__ Vec3 max(const Vec3 &v1, const Vec3 &v2) {
  return Vec3(fmax(v1.e[0], v2.e[0]), fmax(v1.e[1], v2.e[1]), fmax(v1.e[2], v2.e[2]));
}

__host__ __device__ Vec3 min(const Vec3 &v1, const Vec3 &v2) {
  return Vec3(fmin(v1.e[0], v2.e[0]), fmin(v1.e[1], v2.e[1]), fmin(v1.e[2], v2.e[2]));
}

__host__ __device__ Vec3 normalized(const Vec3 &v) {
  float len = length(v);
  return v / (len == 0.f ? 1.f : len);
}
