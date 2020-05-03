#pragma once

#include "math.cuh"
#include "shape.cuh"
#include "bsdf.cuh"

struct Primitive {
  Tri t;
  int bsdf;

  Primitive(const Tri &t, int bsdf = 0) : t(t), bsdf(bsdf) {}
};
