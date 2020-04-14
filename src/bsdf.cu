#include "bsdf.cuh"

void BSDF::update(const Vec3 &n_new, const Vec3 &s_new) {
  n = n_new;
  s = s_new;
  t = cross(n, s);
}

Vec3 BSDF::world2local(const Vec3 &v) const {
  return Vec3(dot(v, n), dot(v, t), dot(v, s));
}

Vec3 BSDF::local2world(const Vec3 &v) const {
  return Vec3(n.e[0] * v.e[0] + t.e[0] * v.e[1] + s.e[0] * v.e[2],
              n.e[1] * v.e[0] + t.e[1] * v.e[1] + s.e[1] * v.e[2],
              n.e[2] * v.e[0] + t.e[2] * v.e[1] + s.e[2] * v.e[2]);
}

Vec3 BSDF::f(const Vec3 &wo_world, const Vec3 &wi_world) const {
  Vec3 wo = world2local(wo_world);
  Vec3 wi = world2local(wi_world);
  if (wo.e[2] == 0) return {0.f, 0.f, 0.f};
  return b->f(wo, wi);
}

Vec3 BSDF::sample_f(const Vec3 &wo_world, Vec3 *wi_world, float u, float v, float *pdf) const {
  Vec3 wo = world2local(wo_world);
  Vec3 wi;
  if (wo.e[2] == 0.f) return {0.f, 0.f, 0.f};
  *pdf = 0.f;

  Vec3 sampled = b->sample_f(wo, &wi, u, v, pdf);

  if (*pdf == 0.f) return {0.f, 0.f, 0.f};
  *wi_world = local2world(wi);

  return sampled;
}

bool BSDF::is_specular() const {
  return b->is_specular();
}
