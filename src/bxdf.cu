#include "bxdf.cuh"
#include "onb_math.cuh"

__device__ Vec3 Lambertian::f(const Vec3 &wo, const Vec3 &wi) const {
  return INV_PI * r;
}

__device__ Vec3 Lambertian::sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, int face, float *pdf_) const {
  *wi = cosine_sample(u, v);
  if (cos_theta(wo) < 0.f) wi->e[2] *= -1.f;
  *pdf_ = pdf(wo, *wi);
  return f(wo, *wi);
}

__device__ float Lambertian::pdf(const Vec3 &wo, const Vec3 &wi) const {
  if (same_hemisphere(wo, wi)) return abscos_theta(wi) * INV_PI;
  return 0.f;
}

__device__ void Lambertian::tex_update(const Vector<Texture> &tex_arr, float u, float v) {
  if (tex_idx < 0) return;
  r = tex_arr[tex_idx].eval(u, v);
}

OrenNayar::OrenNayar(const Vec3 &r, float roughness, int tex) : tex_idx(tex), r(r) {
  float sigma2 = roughness * roughness;
  a = 1.f - sigma2 / (2.f * sigma2 + 0.66f);
  b = 0.45f * sigma2 / (sigma2 + 0.09f);
}

__device__ Vec3 OrenNayar::f(const Vec3 &wo, const Vec3 &wi) const {
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

__device__ Vec3 OrenNayar::sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, int face, float *pdf_) const {
  *wi = cosine_sample(u, v);
  if (cos_theta(wo) < 0.f) wi->e[2] *= -1;
  *pdf_ = pdf(wo, *wi);
  return f(wo, *wi);
}

__device__ float OrenNayar::pdf(const Vec3 &wo, const Vec3 &wi) const {
  return same_hemisphere(wo, wi) ? abscos_theta(wi) * INV_PI : 0.f;
}

__device__ void OrenNayar::tex_update(const Vector<Texture> &tex_arr, float u, float v) {
  if (tex_idx < 0) return;
  r = tex_arr[tex_idx].eval(u, v);
}

__device__ Vec3 AreaLight::f(const Vec3 &wo, const Vec3 &wi) const {
  return INV_PI * r;
}

__device__ Vec3 AreaLight::sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, int face, float *pdf_) const {
  *wi = cosine_sample(u, v);
  if (cos_theta(wo) < 0.f) wi->e[2] *= -1;
  *pdf_ = pdf(wo, *wi);
  return f(wo, *wi);
}

__device__ float AreaLight::pdf(const Vec3 &wo, const Vec3 &wi) const {
  return same_hemisphere(wo, wi) ? abscos_theta(wi) * INV_PI : 0.f;
}

__device__ Vec3 AreaLight::emittance() const {
  return e;
}

__device__ float schlick(float costhetai, float eta1, float eta2) {
  float r0 = (eta1 - eta2) / (eta1 + eta2);
  r0 *= r0;
  costhetai = 1.f - costhetai;
  return r0 + (1.f - r0) * (costhetai * costhetai) * (costhetai * costhetai) * costhetai;
}

__device__ Vec3 TorranceSparrow::f(const Vec3 &wo, const Vec3 &wi) const {
  float costhetao = abscos_theta(wo);
  float costhetai = abscos_theta(wi);
  Vec3 wh = wi + wo;

  if (is_zero(wh) || costhetai == 0.f || costhetao == 0.f) return Vec3(0.f, 0.f, 0.f);

  wh = normalized(wh);

  Vec3 ft = schlick(dot(wi, cos_theta(wh) < 0.f ? (-1 * wh) : wh), eta1, eta2);
  return r * dist.d(wh) * dist.g(wo, wi) * ft / (4.f * costhetai * costhetao);
}

__device__ Vec3 TorranceSparrow::sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, int face, float *pdf) const {
  if (cos_theta(wo) == 0.f) return Vec3(0.f, 0.f, 0.f);
  Vec3 wh = dist.sample_wh(wo, u, v);
  if (dot(wo, wh) < 0.f) return Vec3(0.f, 0.f, 0.f);

  *wi = reflect(wo, wh);
  if (!same_hemisphere(wo, *wi)) return Vec3(0.f, 0.f, 0.f);

  *pdf = dist.pdf(wo, wh) / (4.f * dot(wo, wh));
  return f(wo, *wi);
}

__device__ float TorranceSparrow::pdf(const Vec3 &wo, const Vec3 &wi) const {
  if (!same_hemisphere(wo, wi)) return 0.f;
  Vec3 wh = normalized(wo + wi);
  return dist.pdf(wo, wh) / (4.f * dot(wo, wh));
}

__device__ float fr(float costhetai, float etai, float etat) {
  float sinthetai = sqrtf(fmax(0.f, 1.f - costhetai * costhetai));
  float sinthetat = etai / etat * sinthetai;
  if (sinthetat >= 1) return 1.f;
  float costhetat = sqrtf(fmax(0.f, 1.f - sinthetat * sinthetat));
  float r_parallel = ((etat * costhetai) - (etai * costhetat)) / ((etat * costhetai) + (etai * costhetat));
  float r_perp = ((etai * costhetai) - (etat * costhetat)) / ((etai * costhetai) + (etat * costhetat));
  return ((r_parallel * r_parallel) + (r_perp * r_perp)) / 2.f;
}

__device__ Vec3 SpecularReflection::sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, int face, float *pdf) const {
  *wi = Vec3(-wo.e[0], -wo.e[1], wo.e[2]);
  *pdf = 1.f;
  return r / abscos_theta(*wi);
}

__device__ Vec3 Specular::sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, int face, float *pdf) const {
  bool entering = face == 1.f;

  float etai = entering ? eta1 : eta2;
  float etat = entering ? eta2 : eta1;
  float eta = etai / etat;
  
  float F = fr(cos_theta(wo), etai, etat);
  if (u < F) {
    *wi = Vec3(-wo.e[0], -wo.e[1], wo.e[2]);
    *pdf = F;
    Vec3 ft = F * r;
    return ft / abscos_theta(*wi);
  } else {
    if (!refract(wo, Vec3(0.f, 0.f, 1.f), eta, wi)) return Vec3(0.f);
    *pdf = 1.f - F;
    Vec3 ft = (1.f - F) * t;
    ft *= eta * eta;
    return ft / abscos_theta(*wi);
  }
}

__host__ __device__ BxDFVariant::BxDFVariant(const BxDFVariant &b) {
  which = b.which;
  switch (which) {
    case 1 : lambert = b.lambert;
    case 2 : oren = b.oren;
    case 3 : light = b.light;
    case 4 : microfacet = b.microfacet;
    case 5 : reflect = b.reflect;
    case 6 : specular = b.specular;
  }
}

__host__ __device__ BxDFVariant& BxDFVariant::operator=(const BxDFVariant &b) {
  which = b.which;
  switch (which) {
    case 1 : lambert = b.lambert;
    case 2 : oren = b.oren;
    case 3 : light = b.light;
    case 4 : microfacet = b.microfacet;
    case 5 : reflect = b.reflect;
    case 6 : specular = b.specular;
  }
  return *this;
}

__device__ Vec3 BxDFVariant::f(const Vec3 &wo, const Vec3 &wi) const {
  switch (which) {
    case 1 : return lambert.f(wo, wi);
    case 2 : return oren.f(wo, wi);
    case 3 : return light.f(wo, wi);
    case 4 : return microfacet.f(wo, wi);
    case 5 : return reflect.f(wo, wi);
    case 6 : return specular.f(wo, wi);
  }
  printf("Invalid selector %d!\n", which);
  return Vec3(1.f);
}

__device__ Vec3 BxDFVariant::sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, int face, float *pdf) const {
  switch (which) {
    case 1 : return lambert.sample_f(wo, wi, u, v, face, pdf);
    case 2 : return oren.sample_f(wo, wi, u, v, face, pdf);
    case 3 : return light.sample_f(wo, wi, u, v, face, pdf);
    case 4 : return microfacet.sample_f(wo, wi, u, v, face, pdf);
    case 5 : return reflect.sample_f(wo, wi, u, v, face, pdf);
    case 6 : return specular.sample_f(wo, wi, u, v, face, pdf);
  }
  printf("Invalid selector %d!\n", which);
  *pdf = 0.f;
  return Vec3(1.f);
}

__device__ float BxDFVariant::pdf(const Vec3 &wo, const Vec3 &wi) const {
  switch (which) {
    case 1 : return lambert.pdf(wo, wi);
    case 2 : return oren.pdf(wo, wi);
    case 3 : return light.pdf(wo, wi);
    case 4 : return microfacet.pdf(wo, wi);
    case 5 : return reflect.pdf(wo, wi);
    case 6 : return specular.pdf(wo, wi);
  }
  printf("Invalid selector %d!\n", which);
  return 0.f;
}

__host__ __device__ bool BxDFVariant::is_light() const {
  switch (which) {
    case 1 : return lambert.is_light();
    case 2 : return oren.is_light();
    case 3 : return light.is_light();
    case 4 : return microfacet.is_light();
    case 5 : return reflect.is_light();
    case 6 : return specular.is_light();
  }
  printf("Invalid selector %d!\n", which);
  return false;
}

__device__ Vec3 BxDFVariant::emittance() const {
  switch (which) {
    case 1 : return lambert.emittance();
    case 2 : return oren.emittance();
    case 3 : return light.emittance();
    case 4 : return microfacet.emittance();
    case 5 : return reflect.emittance();
    case 6 : return specular.emittance();
  }
  printf("Invalid selector %d!\n", which);
  return Vec3(0.f);
}

__device__ void BxDFVariant::tex_update(const Vector<Texture> &tex_arr, float u, float v) {
  switch (which) {
    case 1 : return lambert.tex_update(tex_arr, u, v);
    case 2 : return oren.tex_update(tex_arr, u, v);
    case 3 : return light.tex_update(tex_arr, u, v);
    case 4 : return microfacet.tex_update(tex_arr, u, v);
    case 5 : return reflect.tex_update(tex_arr, u, v);
    case 6 : return specular.tex_update(tex_arr, u, v);
  }
  printf("Invalid selector %d!\n", which);
}
