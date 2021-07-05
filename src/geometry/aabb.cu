#include "aabb.cuh"

AABBIntersection AABB::intersects(Ray r) const {
  double tx1 = (lo.x - r.o.x) / r.d.x;
  double tx2 = (hi.x - r.o.x) / r.d.x;

  double tmin = min(tx1, tx2);
  double tmax = max(tx1, tx2);

  double ty1 = (lo.y - r.o.y) / r.d.y;
  double ty2 = (hi.y - r.o.y) / r.d.y;

  tmin = max(tmin, min(ty1, ty2));
  tmax = min(tmax, max(ty1, ty2));

  float time = tmin < 0.f ? tmax : tmin;
  bool hit   = tmax >= tmin && tmax >= 0.f;

  return {time, hit};
}
