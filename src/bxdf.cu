#include "bxdf.cuh"
#include "onb_math.cuh"

Vec3 Lambertian::f(const Vec3 &wo, const Vec3 &wi) const {
  return INV_PI * r;
}

Vec3 Lambertian::sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, float *pdf_) const {
  *wi = cosine_sample(u, v);
  if (cos_theta(wo) < 0.f) wi->e[2] *= -1.f;
  Lambertian t = *((Lambertian *)this);
  *pdf_ = t.pdf(wo, *wi);
  return t.f(wo, *wi);
}

float Lambertian::pdf(const Vec3 &wo, const Vec3 &wi) const {
  return same_hemisphere(wo, wi) ? abscos_theta(wi) * INV_PI : 0.f;
}

void Lambertian::tex_update(const Vector<Texture> &tex_arr, float u, float v) {
  if (tex_idx < 0) return;
  r = tex_arr[tex_idx].eval(u, v);
}

OrenNayar::OrenNayar(const Vec3 &r, float roughness, int tex) : BxDF(tex), r(r) {
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

Vec3 OrenNayar::sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, float *pdf_) const {
  *wi = cosine_sample(u, v);
  if (cos_theta(wo) < 0.f) wi->e[2] *= -1;
  OrenNayar t = *((OrenNayar *)this);
  *pdf_ = t.pdf(wo, *wi);
  return t.f(wo, *wi);
}

float OrenNayar::pdf(const Vec3 &wo, const Vec3 &wi) const {
  return same_hemisphere(wo, wi) ? abscos_theta(wi) * INV_PI : 0.f;
}

void OrenNayar::tex_update(const Vector<Texture> &tex_arr, float u, float v) {
  if (tex_idx < 0) return;
  r = tex_arr[tex_idx].eval(u, v);
}

Vec3 AreaLight::f(const Vec3 &wo, const Vec3 &wi) const {
  return INV_PI * r;
}

Vec3 AreaLight::sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, float *pdf_) const {
  *wi = cosine_sample(u, v);
  if (cos_theta(wo) < 0.f) wi->e[2] *= -1;
  AreaLight t = *((AreaLight *)this);
  *pdf_ = t.pdf(wo, *wi);
  return t.f(wo, *wi);
}

float AreaLight::pdf(const Vec3 &wo, const Vec3 &wi) const {
  return same_hemisphere(wo, wi) ? abscos_theta(wi) * INV_PI : 0.f;
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
  TorranceSparrow t = *((TorranceSparrow *)this);
  return t.f(wo, *wi);
}

float TorranceSparrow::pdf(const Vec3 &wo, const Vec3 &wi) const {
  if (!same_hemisphere(wo, wi)) return 0.f;
  Vec3 wh = normalized(wo + wi);
  return dist->pdf(wo, wh) / (4.f * dot(wo, wh));
}

Vec3 BxDFVariant::f(const Vec3 &wo, const Vec3 &wi) const {
  switch (which) {
    case 1 : return lambert.f(wo, wi);
    case 2 : return oren.f(wo, wi);
    case 3 : return light.f(wo, wi);
    case 4 : return microfacet.f(wo, wi);
  }
  printf("Invalid selector!\n");
  return Vec3(1.f);
}

Vec3 BxDFVariant::sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, float *pdf) const {
  switch (which) {
    case 1 : return lambert.sample_f(wo, wi, u, v, pdf);
    case 2 : return oren.sample_f(wo, wi, u, v, pdf);
    case 3 : return light.sample_f(wo, wi, u, v, pdf);
    case 4 : return microfacet.sample_f(wo, wi, u, v, pdf);
  }
  printf("Invalid selector!\n");
  *pdf = 0.f;
  return Vec3(1.f);
}

float BxDFVariant::pdf(const Vec3 &wo, const Vec3 &wi) const {
  switch (which) {
    case 1 : return lambert.pdf(wo, wi);
    case 2 : return oren.pdf(wo, wi);
    case 3 : return light.pdf(wo, wi);
    case 4 : return microfacet.pdf(wo, wi);
  }
  printf("Invalid selector!\n");
  return 0.f;
}

bool BxDFVariant::is_light() const {
  switch (which) {
    case 1 : return lambert.is_light();
    case 2 : return oren.is_light();
    case 3 : return light.is_light();
    case 4 : return microfacet.is_light();
  }
  printf("Invalid selector!\n");
  return false;
}

Vec3 BxDFVariant::emittance() const {
  switch (which) {
    case 1 : return lambert.emittance();
    case 2 : return oren.emittance();
    case 3 : return light.emittance();
    case 4 : return microfacet.emittance();
  }
  printf("Invalid selector!\n");
  return Vec3(0.f);
}

void BxDFVariant::tex_update(const Vector<Texture> &tex_arr, float u, float v) {
  switch (which) {
    case 1 : return lambert.tex_update(tex_arr, u, v);
    case 2 : return oren.tex_update(tex_arr, u, v);
    case 3 : return light.tex_update(tex_arr, u, v);
    case 4 : return microfacet.tex_update(tex_arr, u, v);
  }
  printf("Invalid selector!\n");
}
