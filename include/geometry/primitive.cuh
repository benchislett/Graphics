#pragma once
#include "ray.cuh"

struct Primitive {
  virtual bool intersects(Ray r) const = 0;
};
