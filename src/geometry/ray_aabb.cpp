#include "geometry.h"

HD HitRecord hit(const Ray& ray, const AABB& bbox) {
  HitRecord record;

  float3 invDir = 1.f / ray.direction;
  float3 t0     = (bbox.lo - ray.origin) * invDir;
  float3 t1     = (bbox.hi - ray.origin) * invDir;

  float tmin = fmaxf(fminf(t0, t1));
  float tmax = fminf(fmaxf(t0, t1));

  record.hit  = (tmax >= 0.f) && (tmin <= tmax);
  record.time = tmin;
  return record;
}
