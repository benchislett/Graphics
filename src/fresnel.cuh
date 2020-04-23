#pragma once

#include "math.cuh"

struct Fresnel {
  const float eta1;
  const float eta2;
  
  Fresnel(float n1, float n2) : eta1(n1), eta2(n2) {}
  float evaluate(float costhetai) const;
};
