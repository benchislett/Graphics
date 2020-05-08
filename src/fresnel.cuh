#pragma once

#include "math.cuh"
#include "cuda.cuh"

struct Fresnel {
  float eta1;
  float eta2;
  
  __host__ __device__ Fresnel(float n1, float n2) : eta1(n1), eta2(n2) {}
  __device__ float evaluate(float costhetai) const;
};
