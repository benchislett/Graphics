#include "intersection.cuh"
#include <cstdio>

#define SIGN(a) (((a) > 0) ? 1 : (((a) < 0) ? -1 : 0))

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

  i->n = normal * (-SIGN(dot(r.d, normal)));
  return true;
}

bool hit(const Ray &r, Tri *tris, int n, Intersection *i) {
  Intersection tmp;
  bool hit_any = false;
  for (int c = 0; c < n; c++) {
    if (hit(r, tris[c], &tmp)) {
      if (hit_any == false || tmp.t < i->t) {
        *i = tmp;
      }
      hit_any = true;
    }
  }
  return hit_any;
}

bool hit_test(const Ray &r, const Slab &s) {
  float xInv = 1.f / r.d.e[0];
  float yInv = 1.f / r.d.e[1];
  float zInv = 1.f / r.d.e[2];

  float t0 = (s.ll.e[0] - r.o.e[0]) * xInv;
  float t1 = (s.ur.e[0] - r.o.e[0]) * xInv;
  float t2 = (s.ll.e[1] - r.o.e[1]) * yInv;
  float t3 = (s.ur.e[1] - r.o.e[1]) * yInv;
  float t4 = (s.ll.e[2] - r.o.e[2]) * zInv;
  float t5 = (s.ur.e[2] - r.o.e[2]) * zInv;

  float tmin = fmax(fmax(fmin(t0, t1), fmin(t2, t3)), fmin(t4, t5));
  float tmax = fmin(fmin(fmax(t0, t1), fmax(t2, t3)), fmax(t4, t5));

  return (0 < tmax) && (tmin < tmax);
}
