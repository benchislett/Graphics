#pragma once

#include "math.cuh"
#include "shape.cuh"
#include "bsdf.cuh"

struct Primitive {
  Tri t;
  BSDF bsdf;

  __host__ __device__ Primitive() {}
  Primitive(const Tri &t, const BSDF &b) : t(t), bsdf(b) {}
};
