#pragma once

#include "helper_math.cuh"

struct Ray {
  float3 origin;
  float3 direction;
};

struct Camera {
  float3 position;

  float3 lower;
  float3 width;
  float3 height;

  Camera(float vfov, float aspect, float3 lookFrom, float3 lookAt, float3 viewUp);

  __host__ __device__ Ray getRay(float u, float v) const;
};
