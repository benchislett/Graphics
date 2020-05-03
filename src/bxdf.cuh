#pragma once

#include "math.cuh"
#include "microfacet.cuh"
#include "fresnel.cuh"
#include "texture.cuh"
#include "vector.cuh"

struct BxDF {
  int tex_idx;

  BxDF(int tex_idx) : tex_idx(tex_idx) {}

  virtual Vec3 f(const Vec3 &wo, const Vec3 &wi) const = 0;
  virtual Vec3 sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, float *pdf) const = 0;
  virtual float pdf(const Vec3 &wo, const Vec3 &wi) const = 0;
  virtual bool is_light() const = 0;
  virtual Vec3 emittance() const = 0;
  virtual void tex_update(const Vector<Texture> &tex_arr, float u, float v) = 0;
};

struct Lambertian : BxDF {
  Vec3 r;

  Lambertian() : BxDF(-1), r(Vec3(1.f)) {}
  Lambertian(const Vec3 &r, int tex = -1) : BxDF(tex), r(r) {}

  Vec3 f(const Vec3 &wo, const Vec3 &wi) const;
  Vec3 sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, float *pdf) const;
  float pdf(const Vec3 &wo, const Vec3 &wi) const;
  bool is_light() const { return false; }
  Vec3 emittance() const { return Vec3(0.f); }
  void tex_update(const Vector<Texture> &tex_arr, float u, float v);
};

struct OrenNayar : BxDF {
  float a, b;
  Vec3 r;

  OrenNayar(const Vec3 &r, float roughness, int tex = -1);
  OrenNayar() : OrenNayar(Vec3(1.f), 1.f, -1) {}

  Vec3 f(const Vec3 &wo, const Vec3 &wi) const;
  Vec3 sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, float *pdf) const;
  float pdf(const Vec3 &wo, const Vec3 &wi) const;
  bool is_light() const { return false; }
  Vec3 emittance() const { return Vec3(0.f); }
  void tex_update(const Vector<Texture> &tex_arr, float u, float v);
};

struct AreaLight : BxDF {
  Vec3 e;
  Vec3 r;

  AreaLight() : BxDF(-1), r(Vec3(1.f)), e(Vec3(10.f)) {}
  AreaLight(const Vec3 &r, const Vec3 &e) : BxDF(-1), r(r), e(e) {}

  Vec3 f(const Vec3 &wo, const Vec3 &wi) const;
  Vec3 sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, float *pdf) const;
  float pdf(const Vec3 &wo, const Vec3 &wi) const;
  bool is_light() const { return true; }
  Vec3 emittance() const;
  void tex_update(const Vector<Texture> &tex_arr, float u, float v) {}
}; 

struct TorranceSparrow : BxDF {
  Vec3 r;
  const MicrofacetDistribution *dist;
  const Fresnel *fresnel;

  TorranceSparrow() : BxDF(-1), r(Vec3(1.f)), dist(NULL), fresnel(NULL) {}
  TorranceSparrow(const Vec3 &r, const MicrofacetDistribution *d, const Fresnel *f) : BxDF(-1), r(r), dist(d), fresnel(f) {}

  Vec3 f(const Vec3 &wo, const Vec3 &wi) const;
  Vec3 sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, float *pdf) const;
  float pdf(const Vec3 &wo, const Vec3 &wi) const;
  bool is_light() const { return false; }
  Vec3 emittance() const { return Vec3(0.f); }
  void tex_update(const Vector<Texture> &tex_arr, float u, float v) {}
};

struct BxDFVariant {
  Lambertian lambert;
  OrenNayar oren;
  AreaLight light;
  TorranceSparrow microfacet;
  int which;

  BxDFVariant(const Lambertian &l) : lambert(l), which(1) {}
  BxDFVariant(const OrenNayar &o) : oren(o), which(2) {}
  BxDFVariant(const AreaLight &li) : light(li), which(3) {}
  BxDFVariant(const TorranceSparrow &ts) : microfacet(ts), which(4) {}

  Vec3 f(const Vec3 &wo, const Vec3 &wi) const;
  Vec3 sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, float *pdf) const;
  float pdf(const Vec3 &wo, const Vec3 &wi) const;
  bool is_light() const;
  Vec3 emittance() const;
  void tex_update(const Vector<Texture> &tex_arr, float u, float v);
};
