#pragma once

#include "math.cuh"
#include "vector.cuh"
#include "bxdf.cuh"

struct BSDF {
  Vec3 n, s, t;
  BxDFVariant b[2];
  int n_bxdfs = 0;

  BSDF(BxDFVariant b1) : b {b1, b1}, n_bxdfs(1) {}
  BSDF(BxDFVariant b1, BxDFVariant b2) : b {b1, b2}, n_bxdfs(2) {}
  BSDF(const BSDF &bs) : n(bs.n), s(bs.s), t(bs.t), b {bs.b[0], bs.b[1]}, n_bxdfs(bs.n_bxdfs) {}

  BSDF& operator=(const BSDF &bs) { n = bs.n; s = bs.s; t = bs.t; b[0] = bs.b[0]; b[1] = bs.b[1]; n_bxdfs = bs.n_bxdfs; return *this; }

  void update(const Vec3 &n, const Vec3 &s, const Vector<Texture> &tex_arr, float u = -1.f, float v = -1.f);

  Vec3 world2local(const Vec3 &v) const;
  Vec3 local2world(const Vec3 &v) const;

  Vec3 f(const Vec3 &wo_world, const Vec3 &wi_world);
  Vec3 sample_f(const Vec3 &wo_world, Vec3 *wi_world, float u, float v, float *pdf, int choice);
  float pdf(const Vec3 &wo_world, const Vec3 &wi_world);
  bool is_light();
  Vec3 emittance();
};
