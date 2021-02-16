#include "camera.cuh"

const float3 view_up = make_float3(0.f, 1.f, 0.f);

Camera make_camera(float3 position, float3 target, float fov, float aspect) {
  float half_height = tanf(fov / 2.f);
  float half_width  = half_height * aspect;

  float3 w = normalized(position - target);
  float3 u = normalized(cross(view_up, w));
  float3 v = cross(w, u);

  float3 horizontal = 2.f * half_width * u;
  float3 vertical   = 2.f * half_height * v;

  float3 project_lower_left = position - (half_width * u) - (half_height * v) - w;

  return (Camera){position, horizontal, vertical, project_lower_left};
}
