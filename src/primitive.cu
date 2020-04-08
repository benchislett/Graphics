#include "primitive.cuh"

Slab bounding_slab(const Slab &s) {
  return { s.ll - 0.001, s.ur + 0.001 };
}

Slab bounding_slab(const Slab &s1, const Slab &s2) {
  return { { fmin(s1.ll.e[0], s2.ll.e[0]), fmin(s1.ll.e[1], s2.ll.e[1]), fmin(s1.ll.e[2], s2.ll.e[2])},
           { fmax(s1.ur.e[0], s2.ur.e[0]), fmax(s1.ur.e[1], s2.ur.e[1]), fmax(s1.ur.e[2], s2.ur.e[2])}};
}

Slab bounding_slab(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
  return { { fmin(fmin(a.e[0], b.e[0]), c.e[0]),
             fmin(fmin(a.e[1], b.e[1]), c.e[1]),
             fmin(fmin(a.e[2], b.e[2]), c.e[2]) },
           { fmax(fmax(a.e[0], b.e[0]), c.e[0]),
             fmax(fmax(a.e[1], b.e[1]), c.e[1]),
             fmax(fmax(a.e[2], b.e[2]), c.e[2]) }};
}

Tri::Tri(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
  Vec3 n = cross(b - a, c - a);
  a = a;
  b = b;
  c = c;
  n_a = n;
  n_b = n;
  n_c = n;
  bound = bounding_slab(a, b, c);
}

Tri::Tri(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3 &n) {
  a = a;
  b = b;
  c = c;
  n_a = n;
  n_b = n;
  n_c = n;
  bound = bounding_slab(a, b, c);
}

Tri::Tri(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3 &n_a, const Vec3 &n_b, const Vec3 &n_c) {
  a = a;
  b = b;
  c = c;
  n_a = n_a;
  n_b = n_b;
  n_c = n_c;
  bound = bounding_slab(a, b, c);
}
