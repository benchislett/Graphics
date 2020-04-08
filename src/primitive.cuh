#pragma once

#include "math.h"

struct Slab {
  Vec3 ll;
  Vec3 ur;

  Slab() {
    ll = { FLT_MAX, FLT_MAX, FLT_MAX };
    ur = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
  }

  void expand(const Slab &s) {
    ll = min(ll, s.ll);
    ur = max(ur, s.ur);
  }
}

Slab bounding_slab(const Slab &s1, const Slab &s2);
Slab bounding_slab(const Vec3 &a, const Vec3 &b, const Vec3 &c);

struct Tri {
  Vec3 a;
  Vec3 b;
  Vec3 c;
  Vec3 n_a;
  Vec3 n_b;
  Vec3 n_c;
  Slab bound;

  Tri(const Vec3 &a, const Vec3 &b, const Vec3 &c);
  Tri(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3 &n);
  Tri(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3 &n_a, const Vec3 &n_b, const Vec3 &n_c);
}
