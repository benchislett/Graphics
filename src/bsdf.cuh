#pragma once

#include "math.cuh"
#include "bxdf.cuh"
#include "texture.cuh"

struct BSDF {
  Vec3 n, s, t;
  BxDF *b[3];
  int n_bxdfs = 0;
  int textures[3];
  int n_textures = 0;

  BSDF(BxDF *b1 = NULL, int t1 = -1, BxDF *b2 = NULL, int t2 = -1, BxDF *b3 = NULL, int t3 = -1) : b {b1, b2, b3}, n_bxdfs(b1 == NULL ? 0 : (b2 == NULL ? 1 : (b3 == NULL ? 2 : 3))), textures {t1, t2, t3}, n_textures(t1 == -1 ? 0 : (t2 == -1 ? 1 : (t3 == -1 ? 2 : 3))) {}

  void update(const Vec3 &n, const Vec3 &s, Texture *tex_arr, float u = -1.f, float v = -1.f);

  Vec3 world2local(const Vec3 &v) const;
  Vec3 local2world(const Vec3 &v) const;

  Vec3 f(const Vec3 &wo_world, const Vec3 &wi_world) const;
  Vec3 sample_f(const Vec3 &wo_world, Vec3 *wi_world, float u, float v, float *pdf, int choice) const;
  float pdf(const Vec3 &wo_world, const Vec3 &wi_world) const;
  bool is_specular() const;
  bool is_light() const;
  Vec3 emittance() const;
};
