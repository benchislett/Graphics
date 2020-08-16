#include "camera.cuh"

#include <cstdio>
Camera::Camera(float vfov, float aspect, float3 lookFrom, float3 lookAt, float3 viewUp) {
  position = lookFrom;

  float halfHeight = tanf(vfov / 2.f);
  float halfWidth = halfHeight * aspect;

  float3 w = normalize(lookFrom - lookAt);
  float3 u = normalize(cross(viewUp, w));
  float3 v = cross(w, u);

  lower = lookFrom - (halfWidth * u) - (halfHeight * v) - w;
  height = 2.f * halfHeight * v;
  width = 2.f * halfWidth * u;
}

__host__ __device__ Ray Camera::getRay(float u, float v) const {
  return {position, lower + (u * width) + (v * height) - position};
}
