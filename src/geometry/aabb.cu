#include "aabb.cuh"

__host__ __device__ AABBIntersection AABB::intersects(Ray r) const {
  double tx1 = (lo.x - r.o.x) / r.d.x;
  double tx2 = (hi.x - r.o.x) / r.d.x;

  double tmin = min(tx1, tx2);
  double tmax = max(tx1, tx2);

  double ty1 = (lo.y - r.o.y) / r.d.y;
  double ty2 = (hi.y - r.o.y) / r.d.y;

  tmin = max(tmin, min(ty1, ty2));
  tmax = min(tmax, max(ty1, ty2));

  float time = tmin < 0.0 ? tmax : tmin;
  bool hit   = tmax >= tmin && tmax >= 0.0;

  return {time, hit};
}

__host__ __device__ AABB AABB::plus(AABB other) const {
  Point3 small = fminf(lo, other.lo);
  Point3 big   = fmaxf(hi, other.hi);

  return AABB(small, big);
}

__host__ __device__ Point3 AABB::centroid() const {
  return (lo + hi) / 2.f;
}

__host__ __device__ Vec3 AABB::length() const {
  return hi - lo;
}

__host__ __device__ float AABB::surface_area() const {
  Vec3 l = length();
  return 2 * (l.x * l.y + l.x * l.z + l.y * l.z);
}
