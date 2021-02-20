#pragma once

#include "cu_math.cuh"
#include "cu_misc.cuh"
#include "onb_math.cuh"

struct BxDFSample {
  float3 out;
  float3 f;
  float pdf;
};

struct DiffuseBRDF {
  float3 albedo;

  IHD float pdf(float3 in, float3 out) {
    if (same_hemisphere(in, out)) {
      return abscos_theta(out) * INV_PI;
    }
    return 0.f;
  }

  IHD float3 f(float3 in, float3 out) {
    return albedo; // * INV_PI;
  }

  IHD BxDFSample sample_f(float3 in, float u, float v) {
    float3 out = cosine_sample(u, v);
    if (cos_theta(in) < 0.f) {
      out.z *= -1.f;
    }
    return {out, f(in, out), pdf(in, out)};
  }
};
