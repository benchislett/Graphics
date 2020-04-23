#include "bxdf.cuh"
#include "onb_math.cuh"

float BxDF::pdf(const Vec3 &wo, const Vec3 &wi) const {
  return same_hemisphere(wo, wi) ? abscos_theta(wi) * INV_PI : 0.f;
}

Vec3 BxDF::emittance() const {
  return Vec3(0.f, 0.f, 0.f);
}

Vec3 Lambertian::f(const Vec3 &wo, const Vec3 &wi) const {
  return INV_PI * r;
}

OrenNayar::OrenNayar(const Vec3 &r, float roughness) : r(r) {
  float sigma2 = roughness * roughness;
  a = 1.f - sigma2 / (2.f * sigma2 + 0.66f);
  b = 0.45f * sigma2 / (sigma2 + 0.09f);
}

Vec3 OrenNayar::f(const Vec3 &wo, const Vec3 &wi) const {
  float sintheta_in = sin_theta(wi);
  float sintheta_out = sin_theta(wo);

  float max_cos = 0.f;
  if (sintheta_in > 0.0001f && sintheta_out > 0.0001f) {
    float sinphi_in = sin_phi(wi);
    float cosphi_in = cos_phi(wi);
    float sinphi_out = sin_phi(wo);
    float cosphi_out = cos_phi(wo);
    float dcos = cosphi_in * cosphi_out + sinphi_in * sinphi_out;
    max_cos = dcos < 0 ? max_cos : dcos;
  }

  float sinalpha, tanbeta;
  if (abscos_theta(wi) > abscos_theta(wo)) {
    sinalpha = sintheta_out;
    tanbeta = sintheta_in / abscos_theta(wi);
  } else {
    sinalpha = sintheta_in;
    tanbeta = sintheta_out / abscos_theta(wo);
  }

  return INV_PI * r * (a + b * max_cos * sinalpha * tanbeta);
}

Vec3 AreaLight::f(const Vec3 &wo, const Vec3 &wi) const {
  return INV_PI * r;
}

Vec3 AreaLight::emittance() const {
  return e;
}
