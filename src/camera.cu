#include "camera.cuh"

Camera::Camera(float vfov, float aspect, const Vec3 &look_from, const Vec3 &look_at, const Vec3 &view_up) {
  pos = look_from;
  Vec3 u, v, w;

  float half_height = tanf(vfov / 2); // vfov should be in radians
  float half_width = half_height * aspect;

  w = normalized(look_from - look_at);
  u = normalized(cross(view_up, w));
  v = cross(w, u);

  lower_left = look_from - (half_width * u) - (half_height * v) - w;

  h = 2 * half_width * u;
  v = 2 * half_height * v;
}

Ray Camera::get_ray(float s, float t) const {
  return { pos, lower_left + (s * h) + (t * v) - pos };
}
