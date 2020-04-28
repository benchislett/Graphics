#pragma once

#include "math.cuh"
#include "microfacet.cuh"
#include "fresnel.cuh"

struct BxDF {
  Vec3 r;

  BxDF(const Vec3 &r) : r(r) {}

  virtual Vec3 f(const Vec3 &wo, const Vec3 &wi) const = 0;
  virtual Vec3 sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, float *pdf) const;
  virtual float pdf(const Vec3 &wi, const Vec3 &wo) const;
  virtual bool is_specular() const = 0;
  virtual bool is_light() const = 0;
  virtual Vec3 emittance() const;
};

struct Lambertian : BxDF {
  Lambertian(const Vec3 &r) : BxDF(r) {}
  Vec3 f(const Vec3 &wo, const Vec3 &wi) const override;
  bool is_specular() const { return false; }
  bool is_light() const { return false; }
};

struct OrenNayar : BxDF {
  float a, b;

  OrenNayar(const Vec3 &r, float roughness);
  Vec3 f(const Vec3 &wo, const Vec3 &wi) const override;
  bool is_specular() const { return false; }
  bool is_light() const { return false; }
};

struct AreaLight : BxDF {
  Vec3 e;

  AreaLight(const Vec3 &r, const Vec3 &e) : BxDF(r), e(e) {}
  Vec3 f(const Vec3 &wo, const Vec3 &wi) const override;
  bool is_specular() const { return false; }
  bool is_light() const { return true; }
  Vec3 emittance() const;
};

struct TorranceSparrow : BxDF {
  const MicrofacetDistribution *dist;
  const Fresnel *fresnel;

  TorranceSparrow(const Vec3 &r, const MicrofacetDistribution *d, const Fresnel *f) :BxDF(r), dist(d), fresnel(f) {}
  Vec3 f(const Vec3 &wo, const Vec3 &wi) const;
  Vec3 sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, float *pdf) const;
  float pdf(const Vec3 &wo, const Vec3 &wi) const;
  bool is_specular() const { return false; }
  bool is_light() const { return false; }
};
