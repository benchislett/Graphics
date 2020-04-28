#include "bxdf.cuh"
#include "onb_math.cuh"

Vec3 BxDF::sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, float *pdf_) const {
  *wi = cosine_sample(u, v);
  if (cos_theta(wo) < 0.f) wi->e[2] *= -1;
  *pdf_ = pdf(wo, *wi);
  return f(wo, *wi);
}

float BxDF::pdf(const Vec3 &wo, const Vec3 &wi) const {
  return same_hemisphere(wo, wi) ? abscos_theta(wi) * INV_PI : 0.f;
}

Vec3 BxDF::emittance() const {
  return Vec3(0.f, 0.f, 0.f);
}

Vec3 Lambertian::f(const Vec3 &wo, const Vec3 &wi) const {
  return INV_PI * r;
}

OrenNayar::OrenNayar(const Vec3 &r, float roughness) : BxDF(r) {
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

Vec3 TorranceSparrow::f(const Vec3 &wo, const Vec3 &wi) const {
  float costhetao = abscos_theta(wo);
  float costhetai = abscos_theta(wi);
  Vec3 wh = wi + wo;

  if (is_zero(wh) || costhetai == 0.f || costhetao == 0.f) return Vec3(0.f, 0.f, 0.f);

  wh = normalized(wh);

  Vec3 f = fresnel->evaluate(dot(wi, cos_theta(wh) < 0.f ? (-1 * wh) : wh));
  return r * dist->d(wh) * dist->g(wo, wi) * f / (4.f * costhetai * costhetao);
}

Vec3 TorranceSparrow::sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, float *pdf) const {
  if (cos_theta(wo) == 0.f) return Vec3(0.f, 0.f, 0.f);
  Vec3 wh = dist->sample_wh(wo, u, v);
  if (dot(wo, wh) < 0.f) return Vec3(0.f, 0.f, 0.f);

  *wi = reflect(wo, wh);
  if (!same_hemisphere(wo, *wi)) return Vec3(0.f, 0.f, 0.f);

  *pdf = dist->pdf(wo, wh) / (4.f * dot(wo, wh));
  return f(wo, *wi);
}

float TorranceSparrow::pdf(const Vec3 &wo, const Vec3 &wi) const {
  if (!same_hemisphere(wo, wi)) return 0.f;
  Vec3 wh = normalized(wo + wi);
  return dist->pdf(wo, wh) / (4.f * dot(wo, wh));
}
