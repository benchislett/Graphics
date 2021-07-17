#pragma once

#include "../geometry/ray.cuh"
#include "../math/float3.cuh"

struct Camera {
  float3 position;
  float3 lower_left;
  float3 vertical;
  float3 horizontal;

  Camera(float vfov, float aspect, float3 look_from, float3 look_at);

  __host__ __device__ Ray get_ray(float u, float v) const;
};
