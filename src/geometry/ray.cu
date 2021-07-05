#include "ray.cuh"

float3 Ray::at(float t) const {
  return o + t * d;
}
