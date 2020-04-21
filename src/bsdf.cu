#include "bsdf.cuh"

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

void BSDF::update(const Vec3 &n_new, const Vec3 &s_new) {
  n = n_new;
  s = s_new;
  t = cross(n, s);
}

Vec3 BSDF::world2local(const Vec3 &v) const {
  return Vec3(dot(v, s), dot(v, t), dot(v, n));
}

Vec3 BSDF::local2world(const Vec3 &v) const {
  return Vec3(s.e[0] * v.e[0] + t.e[0] * v.e[1] + n.e[0] * v.e[2],
              s.e[1] * v.e[0] + t.e[1] * v.e[1] + n.e[1] * v.e[2],
              s.e[2] * v.e[0] + t.e[2] * v.e[1] + n.e[2] * v.e[2]);
}

Vec3 BSDF::f(const Vec3 &wo_world, const Vec3 &wi_world) const {
  Vec3 wo = world2local(wo_world);
  Vec3 wi = world2local(wi_world);
  if (wo.e[2] == 0.f) return {0.f, 0.f, 0.f};
  return b->f(wo, wi);
}

Vec3 BSDF::sample_f(const Vec3 &wo_world, Vec3 *wi_world, float u, float v, float *pdf) const {
  Vec3 wo = world2local(wo_world);
  if (wo.e[2] == 0.f) return {0.f, 0.f, 0.f};

  Vec3 wi = cosine_sample(u, v);
  *wi_world = local2world(cosine_sample(u, v));

  Vec3 sampled = b->f(wo, wi);
  *pdf = b->pdf(wo, wi);

  if (*pdf == 0.f) return {0.f, 0.f, 0.f};

  return sampled;
}

float BSDF::pdf(const Vec3 &wo_world, const Vec3 &wi_world) const {
  Vec3 wo = world2local(wo_world);
  Vec3 wi = world2local(wi_world);
  return b->pdf(wo, wi);
}

bool BSDF::is_specular() const {
  return b->is_specular();
}

bool BSDF::is_light() const {
  return b->is_light();
}

Vec3 BSDF::emittance() const {
  return b->emittance();
}
