#include "intersection.cuh"

// Moller-Trumbore intersection
bool hit(const Ray &r, const Tri &t, Intersection *i) {
  Vec3 edge0 = t.b - t.a;
  Vec3 edge1 = t.c - t.a;

  Vec3 h = cross(r.d, edge1);

  float det = dot(edge0, h);

  if (fabsf(det) < 0.0001f) return false;

  float detInv = 1.0f / det;
  Vec3 s = r.o - t.a;
  float u = detInv * dot(s, h);

  if (u < 0.0f || u > 1.0f) return false;

  Vec3 q = cross(s, edge0);
  float v = detInv * dot(r.d, q);

  if (v < 0.0f || u + v > 1.0f) return false;

  float time = detInv * dot(edge1, q);

  if (time < 0.0001f) return false;

  i->p = r.at(time);
  i->t = time;

#ifdef NO_NORMAL_INTERP
  Vec3 normal = t.n_a;
#else
  float w = 1.f - u - v;
  Vec3 normal = t.n_a * w + t.n_b * u + t.n_c * v;
#endif

  h->n = normal * (-std::sgn(dot(r.d, normal)));

  return true;
}
