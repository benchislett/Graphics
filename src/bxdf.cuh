#pragma once

#include "math.cuh"
#include "microfacet.cuh"
#include "fresnel.cuh"

struct BxDF {
  virtual Vec3 f(const Vec3 &wo, const Vec3 &wi) const = 0;
  virtual float pdf(const Vec3 &wi, const Vec3 &wo) const;
  virtual bool is_specular() const = 0;
  virtual bool is_light() const = 0;
  virtual Vec3 emittance() const;
};

struct Lambertian : BxDF {
  Vec3 r;

  Lambertian(const Vec3 &r) : r(r) {}
  Vec3 f(const Vec3 &wo, const Vec3 &wi) const override;
  bool is_specular() const { return false; }
  bool is_light() const { return false; }
};

struct OrenNayar : BxDF {
  float a, b;
  Vec3 r;

  OrenNayar(const Vec3 &r, float roughness);
  Vec3 f(const Vec3 &wo, const Vec3 &wi) const override;
  bool is_specular() const { return false; }
  bool is_light() const { return false; }
};

struct AreaLight : BxDF {
  Vec3 r;
  Vec3 e;

  AreaLight(const Vec3 &r, const Vec3 &e) : r(r), e(e) {}
  Vec3 f(const Vec3 &wo, const Vec3 &wi) const override;
  bool is_specular() const { return false; }
  bool is_light() const { return true; }
  Vec3 emittance() const;
};

struct TorranceSparrow : BxDF {
  const Vec3 r;
  const MicrofacetDistribution *dist;
  const Fresnel *fresnel;

  TorranceSparrow(const Vec3 &r, const MicrofacetDistribution *d, const Fresnel *f) : r(r), dist(d), fresnel(f) {}

  Vec3 f(const Vec3 &wo, const Vec3 &wi) const;
  Vec3 sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, float *pdf) const;
  float pdf(const Vec3 &wo, const Vec3 &wi) const;
  bool is_specular() const { return false; }
  bool is_light() const { return false; }
};
