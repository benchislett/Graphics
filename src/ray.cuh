#pragma once

#include "math.cuh"

struct Ray {
  Vec3 o;
  Vec3 d;

  Ray(const Vec3 &o, const Vec3 &d) : o(o), d(d) {}

  Vec3 at(float t) const;
};
