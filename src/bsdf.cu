#include "bsdf.cuh"

__device__ void BSDF::update(const Vec3 &n_new, const Vec3 &s_new, const Vector<Texture> &tex_arr, float u, float v) {
  n = n_new;
  s = s_new;
  t = cross(n, s);

  for (int i = 0; i < n_bxdfs; i++) {
    b[i].tex_update(tex_arr, u, v);
  }
}

__device__ Vec3 BSDF::world2local(const Vec3 &v) const {
  return Vec3(dot(v, s), dot(v, t), dot(v, n));
}

__device__ Vec3 BSDF::local2world(const Vec3 &v) const {
  return Vec3(s.e[0] * v.e[0] + t.e[0] * v.e[1] + n.e[0] * v.e[2],
              s.e[1] * v.e[0] + t.e[1] * v.e[1] + n.e[1] * v.e[2],
              s.e[2] * v.e[0] + t.e[2] * v.e[1] + n.e[2] * v.e[2]);
}

__device__ Vec3 BSDF::f(const Vec3 &wo_world, const Vec3 &wi_world) const {
  Vec3 wo = world2local(wo_world);
  Vec3 wi = world2local(wi_world);
  if (wo.e[2] == 0.f) return {0.f, 0.f, 0.f};

  Vec3 val(0.f, 0.f, 0.f);
  for (int i = 0; i < n_bxdfs; i++) {
    val += b[i].f(wo, wi);
  }
  return val;
}

__device__ Vec3 BSDF::sample_f(const Vec3 &wo_world, Vec3 *wi_world, float u, float v, float *pdf, int choice) const {
  Vec3 wo = world2local(wo_world);
  if (wo.e[2] == 0.f) return {0.f, 0.f, 0.f};

  Vec3 wi;
  Vec3 val = b[choice].sample_f(wo, &wi, u, v, pdf);

  *wi_world = local2world(wi);

  for (int i = 0; i < n_bxdfs; i++) {
    if (i != choice) {
      (*pdf) += b[i].pdf(wo, wi);
      val += b[i].f(wo, wi);
    }
  }

  if (*pdf == 0.f) return {0.f, 0.f, 0.f};

  return val;
}

__device__ float BSDF::pdf(const Vec3 &wo_world, const Vec3 &wi_world) const {
  Vec3 wo = world2local(wo_world);
  Vec3 wi = world2local(wi_world);
  float pdf = 0.f;
  for (int i = 0; i < n_bxdfs; i++) {
    pdf += b[i].pdf(wo, wi);
  }
  return pdf;
}

__host__ __device__ bool BSDF::is_light() const {
  for (int i = 0; i < n_bxdfs; i++) {
    if (b[i].is_light()){
      return true;
    }
  }
  return false;
}

__device__ Vec3 BSDF::emittance() const {
  Vec3 emit(0.f, 0.f, 0.f);
  for (int i = 0; i < n_bxdfs; i++) {
    emit += b[i].emittance();
  }
  return emit;
}
