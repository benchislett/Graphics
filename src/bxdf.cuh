#pragma once

#include "math.cuh"

struct BxDF {
  virtual Vec3 f(const Vec3 &wo, const Vec3 &wi) const = 0;
  virtual float pdf(const Vec3 &wi, const Vec3 &wo) const;
  virtual bool is_specular() const = 0;
};

struct OrenNayar : BxDF {
  float a, b;
  Vec3 r;

  OrenNayar(const Vec3 &r, float roughness);
  Vec3 f(const Vec3 &wo, const Vec3 &wi) const override;
  bool is_specular() const;
};
