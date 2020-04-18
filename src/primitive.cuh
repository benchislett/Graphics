#pragma once

#include "math.cuh"
#include "shape.cuh"
#include "bsdf.cuh"

struct Primitive {
  Tri t;
  BSDF *bsdf;

  Primitive(const Tri &t);
  Primitive(const Tri &t, BSDF *b) : t(t), bsdf(b) {}
};
