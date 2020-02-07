#include "rt.cuh"

__host__ __device__ bool hit(const Ray &r, const AABB &s, HitData *h)
{
  // Expanding components because this needs to be VERY fast
  float x = 1.0f / r.d.x;
  float y = 1.0f / r.d.y;
  float z = 1.0f / r.d.z;

  float t1 = (s.ll.x - r.from.x) * x;
  float t2 = (s.ur.x - r.from.x) * x;
  float t3 = (s.ll.y - r.from.y) * y;
  float t4 = (s.ur.y - r.from.y) * y;
  float t5 = (s.ll.z - r.from.z) * z;
  float t6 = (s.ur.z - r.from.z) * z;

  float tmin = MAX(MAX(MIN(t1, t2), MIN(t3, t4)), MIN(t5, t6));
  float tmax = MIN(MIN(MAX(t1, t2), MAX(t3, t4)), MAX(t5, t6));

  return (0 < tmax) && (tmin < tmax);
}


