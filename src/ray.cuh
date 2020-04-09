#pragma once

#include "math.cuh"

struct Ray {
  Vec3 o;
  Vec3 d;

  Vec3 at(float t) const;
};
