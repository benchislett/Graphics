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

struct Tri {
  Vec3 a;
  Vec3 b;
  Vec3 c;
  Vec3 n_a;
  Vec3 n_b;
  Vec3 n_c;
  Slab bound;
}
