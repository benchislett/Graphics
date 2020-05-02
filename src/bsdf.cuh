#pragma once

#include "math.cuh"
#include "vector.cuh"
#include "bxdf.cuh"

struct BSDF {
  Vec3 n, s, t;
  BxDF *b[3];
  int n_bxdfs = 0;

  BSDF(BxDF *b1 = NULL, BxDF *b2 = NULL, BxDF *b3 = NULL) : b {b1, b2, b3}, n_bxdfs(b1 == NULL ? 0 : (b2 == NULL ? 1 : (b3 == NULL ? 2 : 3))) {}

  void update(const Vec3 &n, const Vec3 &s, const Vector<Texture> &tex_arr, float u = -1.f, float v = -1.f);

  Vec3 world2local(const Vec3 &v) const;
  Vec3 local2world(const Vec3 &v) const;

  Vec3 f(const Vec3 &wo_world, const Vec3 &wi_world) const;
  Vec3 sample_f(const Vec3 &wo_world, Vec3 *wi_world, float u, float v, float *pdf, int choice) const;
  float pdf(const Vec3 &wo_world, const Vec3 &wi_world) const;
  bool is_specular() const;
  bool is_light() const;
  Vec3 emittance() const;
};
