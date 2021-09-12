#include "float3.cuh"
#include "ray.cuh"
#include "sphere.cuh"

__host__ __device__ SphereIntersection Sphere::intersects(Ray r) const {
  SphereIntersection no_hit = {};

  float radius2 = radius * radius;
  Point3 L      = center - r.o;
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

  Point3 isect_point = r.at(t);
  Vec3 normal        = normalized(isect_point - center);

  return {normal, isect_point, t, true};
}
