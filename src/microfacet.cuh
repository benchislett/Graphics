#pragma once

#include "math.cuh"

struct MicrofacetDistribution {
  const bool sample_visible_area;

  MicrofacetDistribution(bool s) : sample_visible_area(s) {}

  virtual Vec3 sample_wh(const Vec3 &wo, float u, float v) const = 0;
  virtual float d(const Vec3 &wh) const = 0;
  virtual float lambda(const Vec3 &w) const = 0;
  float g1(const Vec3 &w) const;
  float g(const Vec3 &wo, const Vec3 &wi) const;
  float pdf(const Vec3 &wo, const Vec3 &wh) const;
};

struct Beckmann : MicrofacetDistribution {
  float alpha_x, alpha_y;

  Beckmann(float ax, float ay, bool s = true) : MicrofacetDistribution(s), alpha_x(ax), alpha_y(ay) {}
  Beckmann(float roughness, bool s = true);

  float lambda(const Vec3 &w) const;
  Vec3 sample_wh(const Vec3 &wo, float u, float v) const;
  float d(const Vec3 &wh) const;
};
