#pragma once

#include "cu_math.h"
#include "geometry.h"

struct Camera {
  float3 position;

  float3 horizontal;
  float3 vertical;
  float3 project_lower_left;
};

// Expects fov in radians
Camera make_camera(float3 position, float3 target, float fov, float aspect);

HD Ray get_ray(const Camera camera, float u, float v);
