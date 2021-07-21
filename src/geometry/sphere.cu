#include "float3.cuh"
#include "ray.cuh"
#include "sphere.cuh"

__host__ __device__ SphereIntersection Sphere::intersects(Ray r) const {
  SphereIntersection no_hit = {};

  float radius2 = radius * radius;
  float3 L      = center - r.o;
  float tca     = dot(L, r.d);
  float d2      = dot(L, L) - tca * tca;
  if (d2 > radius2)
    return no_hit;

  float thc  = sqrtf(radius2 - d2);
  float tmin = tca - thc;
  float tmax = tca + thc;

  float t = tmin;
  if (tmax < 0)
    return no_hit;
  else if (tmin < 0)
    t = tmax;

  float3 point  = r.at(t);
  float3 normal = normalized(point - center);

  return {normal, point, t, true};
}
