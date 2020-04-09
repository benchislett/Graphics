#pragma once

#include "math.cuh"
#include "ray.cuh"

struct Camera {
  Vec3 pos;
  Vec3 lower_left;
  Vec3 h;
  Vec3 v;

  Camera(float vfov, float aspect, const Vec3 &look_from, const Vec3 &look_at, const Vec3 &view_up);

  Ray get_ray(float s, float t) const;
};
