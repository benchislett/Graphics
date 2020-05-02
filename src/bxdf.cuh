#pragma once

#include "math.cuh"
#include "microfacet.cuh"
#include "fresnel.cuh"
#include "texture.cuh"

struct BxDF {
  int tex_idx;

  BxDF(int tex_idx) : tex_idx(tex_idx) {}

  virtual Vec3 f(const Vec3 &wo, const Vec3 &wi) const = 0;
  virtual Vec3 sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, float *pdf) const;
  virtual float pdf(const Vec3 &wi, const Vec3 &wo) const;
  virtual bool is_specular() const { return false; }
  virtual bool is_light() const { return false; }
  virtual Vec3 emittance() const { return Vec3(0.f); }
  virtual void tex_update(Texture *tex_arr, float u, float v);
};

struct Lambertian : BxDF {
  Vec3 r;

  Lambertian(const Vec3 &r, int tex = -1) : BxDF(tex), r(r) {}
  Vec3 f(const Vec3 &wo, const Vec3 &wi) const;
  void tex_update(Texture *tex_arr, float u, float v);
};

struct OrenNayar : BxDF {
  float a, b;
  Vec3 r;

  OrenNayar(const Vec3 &r, float roughness, int tex = -1);
  Vec3 f(const Vec3 &wo, const Vec3 &wi) const;
  void tex_update(Texture *tex_arr, float u, float v);
};

struct AreaLight : BxDF {
  Vec3 e;
  Vec3 r;

  AreaLight(const Vec3 &r, const Vec3 &e) : BxDF(-1), r(r), e(e) {}
  Vec3 f(const Vec3 &wo, const Vec3 &wi) const;
  bool is_light() const { return true; }
  Vec3 emittance() const;
};

struct TorranceSparrow : BxDF {
  Vec3 r;
  const MicrofacetDistribution *dist;
  const Fresnel *fresnel;

  TorranceSparrow(const Vec3 &r, const MicrofacetDistribution *d, const Fresnel *f) : BxDF(-1), r(r), dist(d), fresnel(f) {}
  Vec3 f(const Vec3 &wo, const Vec3 &wi) const;
  Vec3 sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, float *pdf) const;
  float pdf(const Vec3 &wo, const Vec3 &wi) const;
};
