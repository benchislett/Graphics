#include "camera.cuh"

constexpr Vec3 view_up(0.0, -1.0, 0.0);

Camera::Camera(float vfov, float aspect, Point3 look_from, Point3 look_at) {
  position = look_from;
  Vec3 u_, v_, w_;

  // vfov should be in radians
  float height = 2.0 * tanf(vfov / 2.f);
  float width  = height * aspect;

  w_ = normalized(look_from - look_at);
  u_ = normalized(cross(view_up, w_));
  v_ = cross(w_, u_);

  lower_left = look_from - (width * 0.5 * u_) - (height * 0.5 * v_) - w_;

  vertical   = v_ * height;
  horizontal = u_ * width;
}

__host__ __device__ Ray Camera::get_ray(float u, float v) const {
  return Ray(position, lower_left + (u * horizontal) + (v * vertical) - position);
}
