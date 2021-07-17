#include "ray.cuh"

__host__ __device__ float3 Ray::at(float t) const {
  return o + t * d;
}
