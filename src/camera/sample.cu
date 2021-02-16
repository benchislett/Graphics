#include "camera.cuh"

HD Ray get_ray(const Camera camera, float u, float v) {
  v                = 1.f - v;
  float3 x         = u * camera.horizontal;
  float3 y         = v * camera.vertical;
  float3 direction = camera.project_lower_left - camera.position;
  return (Ray){camera.position, normalized(direction + x + y)};
}
