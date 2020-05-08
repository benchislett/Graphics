#pragma once

#include "math.cuh"
#include "cuda.cuh"
#include "microfacet.cuh"
#include "fresnel.cuh"
#include "texture.cuh"
#include "vector.cuh"

struct Lambertian {
  int tex_idx;
  Vec3 r;

  __host__ __device__ Lambertian() : tex_idx(-1), r(Vec3(1.f)) {}
  Lambertian(const Vec3 &r, int tex = -1) : tex_idx(tex), r(r) {}

  __device__ Vec3 f(const Vec3 &wo, const Vec3 &wi) const;
  __device__ Vec3 sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, float *pdf) const;
  __device__ float pdf(const Vec3 &wo, const Vec3 &wi) const;
  __host__ __device__ bool is_light() const { return false; }
  __device__ Vec3 emittance() const { return Vec3(0.f); }
  __device__ void tex_update(const Vector<Texture> &tex_arr, float u, float v);
};

struct OrenNayar {
  int tex_idx;
  float a, b;
  Vec3 r;

  __host__ __device__ OrenNayar() : tex_idx(-1), a(1.f), b(0.f), r(Vec3(0.f)) {}
  OrenNayar(const Vec3 &r, float roughness, int tex = -1);

  __device__ Vec3 f(const Vec3 &wo, const Vec3 &wi) const;
  __device__ Vec3 sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, float *pdf) const;
  __device__ float pdf(const Vec3 &wo, const Vec3 &wi) const;
  __host__ __device__ bool is_light() const { return false; }
  __device__ Vec3 emittance() const { return Vec3(0.f); }
  __device__ void tex_update(const Vector<Texture> &tex_arr, float u, float v);
};

struct AreaLight {
  Vec3 e;
  Vec3 r;

  __host__ __device__ AreaLight() : r(Vec3(1.f)), e(Vec3(10.f)) {}
  AreaLight(const Vec3 &r, const Vec3 &e) : r(r), e(e) {}

  __device__ Vec3 f(const Vec3 &wo, const Vec3 &wi) const;
  __device__ Vec3 sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, float *pdf) const;
  __device__ float pdf(const Vec3 &wo, const Vec3 &wi) const;
  __host__ __device__ bool is_light() const { return true; }
  __device__ Vec3 emittance() const;
  __device__ void tex_update(const Vector<Texture> &tex_arr, float u, float v) {}
}; 

struct TorranceSparrow {
  Vec3 r;
  BeckmannDistribution dist;
  Fresnel fresnel;

  __host__ __device__ TorranceSparrow() : r(Vec3(1.f)), dist { 0.f, 0.f }, fresnel { 1.f, 1.5 } {}
  TorranceSparrow(const Vec3 &r, const BeckmannDistribution d, const Fresnel f) : r(r), dist(d), fresnel(f) {}

  __device__ Vec3 f(const Vec3 &wo, const Vec3 &wi) const;
  __device__ Vec3 sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, float *pdf) const;
  __device__ float pdf(const Vec3 &wo, const Vec3 &wi) const;
  __host__ __device__ bool is_light() const { return false; }
  __device__ Vec3 emittance() const { return Vec3(0.f); }
  __device__ void tex_update(const Vector<Texture> &tex_arr, float u, float v) {}
};

struct BxDFVariant {
  union {
    Lambertian lambert;
    OrenNayar oren;
    AreaLight light;
    TorranceSparrow microfacet;
  };
  int which;

  __host__ __device__ BxDFVariant() : which(-1) {}
  BxDFVariant(const Lambertian &l) : lambert(l), which(1) {}
  BxDFVariant(const OrenNayar &o) : oren(o), which(2) {}
  BxDFVariant(const AreaLight &li) : light(li), which(3) {}
  BxDFVariant(const TorranceSparrow &ts) : microfacet(ts), which(4) {}
  __host__ __device__ BxDFVariant(const BxDFVariant &b);

  __host__ __device__ BxDFVariant& operator=(const BxDFVariant &b);

  __device__ Vec3 f(const Vec3 &wo, const Vec3 &wi) const;
  __device__ Vec3 sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, float *pdf) const;
  __device__ float pdf(const Vec3 &wo, const Vec3 &wi) const;
  __host__ __device__ bool is_light() const;
  __device__ Vec3 emittance() const;
  __device__ void tex_update(const Vector<Texture> &tex_arr, float u, float v);
};
