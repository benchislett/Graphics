#pragma once

#include "math.cuh"
#include "bxdf.cuh"

struct BSDF {
  Vec3 n, s, t;
  BxDF *b;

  BSDF() : b(NULL) {}
  BSDF(BxDF *b) : b(b) {}

  void update(const Vec3 &n, const Vec3 &s);

  Vec3 world2local(const Vec3 &v) const;
  Vec3 local2world(const Vec3 &v) const;

  Vec3 f(const Vec3 &wo_world, const Vec3 &wi_world) const;
  Vec3 sample_f(const Vec3 &wo_world, Vec3 *wi_world, float u, float v, float *pdf) const;
  bool is_specular() const;
};
