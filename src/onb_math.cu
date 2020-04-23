#include "onb_math.cuh"

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

void concentric_sample_disk(float *u, float *v) {
  float uu = 2.f * (*u) - 1.f;
  float vv = 2.f * (*v) - 1.f;

  if (uu == 0.f || vv == 0.f) {
    *u = 0.f;
    *v = 0.f;
    return;
  }

  float r, theta;
  if (fabs(uu) > fabs(vv)) {
    r = uu;
    theta = PI_OVER_4 * (vv / uu);
  } else {
    r = vv;
    theta = PI_OVER_2 - PI_OVER_4 * (uu / vv);
  }

  *u = r * cosf(theta);
  *v = r * sinf(theta);
}

Vec3 cosine_sample(float u, float v) {
  concentric_sample_disk(&u, &v);

  float z = sqrtf(fmax(0.f, 1.f - u * u - v * v));
  return Vec3(u, v, z);
}

/*
Vec3 cosine_sample(float u, float v) {
  float phi = 2.f * PI * u;
  float v_sqrt = sqrtf(v);
  float x = cosf(phi) * v_sqrt;
  float y = sinf(phi) * v_sqrt;
  float z = sqrtf(1.f - v);
  return Vec3(x, y, z);
}
*/
