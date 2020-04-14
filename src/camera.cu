#include "camera.cuh"

#include <cstdio>

Camera::Camera(float vfov, float aspect, const Vec3 &look_from, const Vec3 &look_at, const Vec3 &view_up) {
  pos = look_from;
  Vec3 u_, v_, w_;

  float half_height = tanf(vfov / 2.f); // vfov should be in radians
  float half_width = half_height * aspect;

  w_ = normalized(look_from - look_at);
  u_ = normalized(cross(view_up, w_));
  v_ = cross(w_, u_);

  lower_left = look_from - (half_width * u_) - (half_height * v_) - w_;

  h = 2 * half_width * u_;
  v = 2 * half_height * v_;
}

Ray Camera::get_ray(float s, float t) const {
  return Ray(pos, lower_left + (s * h) + (t * v) - pos);
}
