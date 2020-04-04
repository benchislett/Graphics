#pragma once

#include "math.cuh"

struct Ray {
  Vec3 o;
  Vec3 d;

  Vec3 at(float t) const { return Vec3(o.e[0] + t * d.e[0], o.e[1] + t * d.e[1], o.e[2] + t * d.e[2]); }
}
