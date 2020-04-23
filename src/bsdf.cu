#include "bsdf.cuh"

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

  Vec3 wi;
  Vec3 sampled = b->sample_f(wo, &wi, u, v, pdf);
  *wi_world = local2world(wi);

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
