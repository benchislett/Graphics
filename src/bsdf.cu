#include "bsdf.cuh"

Vec3 cosine_sample(float u, float v) {
  float phi = 2.f * PI * u;
  float v_sqrt = sqrtf(v);
  float x = cosf(phi) * v_sqrt;
  float y = sinf(phi) * v_sqrt;
  float z = sqrtf(1.f - v);
  return Vec3(x, y, z);
}

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
  if (wo.e[2] == 0) return {0.f, 0.f, 0.f};
  return b->f(wo, wi);
}

Vec3 BSDF::sample_f(const Vec3 &wo_world, Vec3 *wi_world, float u, float v, float *pdf) const {
  Vec3 wo = world2local(wo_world);
  if (wo.e[2] == 0.f) return {0.f, 0.f, 0.f};

  *wi_world = cosine_sample(u, v);
  Vec3 wi = world2local(*wi_world);

  Vec3 sampled = b->f(wo, wi);
  *pdf = b->pdf(wo, wi);

  if (*pdf == 0.f) return {0.f, 0.f, 0.f};

  return sampled;
}

bool BSDF::is_specular() const {
  return b->is_specular();
}
