#include "ray.cuh"

__host__ __device__ Point3 Ray::at(float t) const {
  return o + t * d;
}
