#include "ray.cuh"

__device__ Vec3 Ray::at(float t) const {
  return Vec3(o.e[0] + t * d.e[0], o.e[1] + t * d.e[1], o.e[2] + t * d.e[2]);
}
