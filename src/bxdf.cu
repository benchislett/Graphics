#include "bxdf.cuh"

float cos_theta(const Vec3 &w) {
  return w.e[2];
}

float cos2_theta(const Vec3 &w) {
  return w.e[2] * w.e[2];
}

float abscos_theta(const Vec3 &w) {
  return w.e[2] < 0.f ? -w.e[2] : w.e[2];
}

float sin2_theta(const Vec3 &w) {
  return 1.f - cos2_theta(w);
}

float sin_theta(const Vec3 &w) {
  return sqrtf(sin2_theta(w));
}

float tan_theta(const Vec3 &w) {
  return sin_theta(w) / cos_theta(w);
}

float tan2_theta(const Vec3 &w) {
  return sin2_theta(w) / cos2_theta(w);
}

float cos_phi(const Vec3 &w) {
  float sintheta = sin_theta(w);
  float x = (sintheta == 0.f) ? 0.f : w.e[0] / sintheta;
  return (x < -1.f ? -1.f : (x > 1.f ? 1.f : x));
}

float sin_phi(const Vec3 &w) {
  float sintheta = sin_theta(w);
  float x = (sintheta == 0.f) ? 0.f : w.e[1] / sintheta;
  return (x < -1.f ? -1.f : (x > 1.f ? 1.f : x));
}

float cos2_phi(const Vec3 &w) {
  float x = cos_phi(w);
  return x * x;
}

float sin2_phi(const Vec3 &w) {
  float x = sin_phi(w);
  return x * x;
}

Vec3 reflect(const Vec3 &w, const Vec3 &n) {
  return 2 * dot(w, n) * n - w;
}

bool refract(const Vec3 &w_in, const Vec3 &n, float eta, Vec3 *w_t) {
  float costheta_in = dot(n, w_in);
  float sin2theta_in = 1.f - costheta_in * costheta_in;
  sin2theta_in = sin2theta_in < 0.f ? 0.f : sin2theta_in;
  float sin2theta_t = eta * eta * sin2theta_in;

  if (sin2theta_t >= 1) return false;
  float costheta_t = sqrtf(1.f - sin2theta_t);
  *w_t = -eta * w_in + (eta * costheta_in - costheta_t) * n;
  return true;
}

bool same_hemisphere(const Vec3 &w, const Vec3 &wp) {
  return w.e[2] * wp.e[2] > 0.f;
}

void concentricsample_disk(float *u, float *v) {
  float x = 2.f * (*u) - 1.f;
  float y = 2.f * (*v) - 1.f;

  if (x == 0.f && y == 0.f) {
    *u = 0.f;
    *v = 0.f;
    return;
  }

  float theta, r;
  if (std::abs(x) > std::abs(y)) {
    r = x;
    theta = PI_OVER_4 * (y / x);
  } else {
    r = y;
    theta = PI_OVER_2 - PI_OVER_4 * (x / y);
  }
  
  *u = r * cosf(theta);
  *v = r * sinf(theta);
}

Vec3 cosinesample_hemisphere(float u, float v) {
  concentricsample_disk(&u, &v);

  float z = 1.f - u * u - v * v;
  z = z < 0.f ? 0.f : sqrtf(z);
  return Vec3(u, v, z);
}

Vec3 BxDF::sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, float *ret_pdf) const {
  *wi = cosinesample_hemisphere(u, v);
  if (wo.e[2] < 0.f) wi->e[2] = -wi->e[2];
  *ret_pdf = pdf(wo, *wi);
  return f(wo, *wi);
}

float BxDF::pdf(const Vec3 &wo, const Vec3 &wi) const {
  return same_hemisphere(wo, wi) ? abscos_theta(wi) * INV_PI : 0.f;
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

bool OrenNayar::is_specular() const {
  return false;
}