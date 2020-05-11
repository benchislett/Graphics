#pragma once

#include "math.cuh"
#include "cuda.cuh"
#include "microfacet.cuh"
#include "texture.cuh"
#include "vector.cuh"

struct Lambertian {
  int tex_idx;
  Vec3 r;

  __host__ __device__ Lambertian() {}
  Lambertian(const Vec3 &r, int tex = -1) : tex_idx(tex), r(r) {}

  __device__ Vec3 f(const Vec3 &wo, const Vec3 &wi) const;
  __device__ Vec3 sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, int face, float *pdf) const;
  __device__ float pdf(const Vec3 &wo, const Vec3 &wi) const;
  __host__ __device__ bool is_light() const { return false; }
  __device__ Vec3 emittance() const { return Vec3(0.f); }
  __device__ void tex_update(const Vector<Texture> &tex_arr, float u, float v);
};

struct OrenNayar {
  int tex_idx;
  float a, b;
  Vec3 r;

  __host__ __device__ OrenNayar() {}
  OrenNayar(const Vec3 &r, float roughness, int tex = -1);

  __device__ Vec3 f(const Vec3 &wo, const Vec3 &wi) const;
  __device__ Vec3 sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, int face, float *pdf) const;
  __device__ float pdf(const Vec3 &wo, const Vec3 &wi) const;
  __host__ __device__ bool is_light() const { return false; }
  __device__ Vec3 emittance() const { return Vec3(0.f); }
  __device__ void tex_update(const Vector<Texture> &tex_arr, float u, float v);
};

struct AreaLight {
  Vec3 e;
  Vec3 r;

  __host__ __device__ AreaLight() {}
  AreaLight(const Vec3 &r, const Vec3 &e) : r(r), e(e) {}

  __device__ Vec3 f(const Vec3 &wo, const Vec3 &wi) const;
  __device__ Vec3 sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, int face, float *pdf) const;
  __device__ float pdf(const Vec3 &wo, const Vec3 &wi) const;
  __host__ __device__ bool is_light() const { return true; }
  __device__ Vec3 emittance() const;
  __device__ void tex_update(const Vector<Texture> &tex_arr, float u, float v) {}
}; 

struct TorranceSparrow {
  Vec3 r;
  BeckmannDistribution dist;
  float eta1, eta2;

  __host__ __device__ TorranceSparrow() {}
  TorranceSparrow(const Vec3 &r, const BeckmannDistribution &d, float n1, float n2) : r(r), dist(d), eta1(n1), eta2(n2) {}

  __device__ Vec3 f(const Vec3 &wo, const Vec3 &wi) const;
  __device__ Vec3 sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, int face, float *pdf) const;
  __device__ float pdf(const Vec3 &wo, const Vec3 &wi) const;
  __host__ __device__ bool is_light() const { return false; }
  __device__ Vec3 emittance() const { return Vec3(0.f); }
  __device__ void tex_update(const Vector<Texture> &tex_arr, float u, float v) {}
};

struct SpecularReflection {
  Vec3 r;
  float eta1, eta2;

  __host__ __device__ SpecularReflection() {}
  SpecularReflection(const Vec3 &r, float n1, float n2) : r(r), eta1(n1), eta2(n2) {}

  __device__ Vec3 f(const Vec3 &wo, const Vec3 &wi) const { return Vec3(0.f); }
  __device__ Vec3 sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, int face, float *pdf) const;
  __device__ float pdf(const Vec3 &wo, const Vec3 &wi) const { return 0.f; }
  __host__ __device__ bool is_light() const { return false; }
  __device__ Vec3 emittance() const { return Vec3(0.f); }
  __device__ void tex_update(const Vector<Texture> &tex_arr, float u, float v) {}
};

struct Specular {
  Vec3 r, t;
  float eta1, eta2;

  __host__ __device__ Specular() {}
  Specular(const Vec3 &r, const Vec3 &t, float n1, float n2) : r(r), t(t), eta1(n1), eta2(n2) {}

  __device__ Vec3 f(const Vec3 &wo, const Vec3 &wi) const { return Vec3(0.f); }
  __device__ Vec3 sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, int face, float *pdf) const;
  __device__ float pdf(const Vec3 &wo, const Vec3 &wi) const { return 0.f; }
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
    SpecularReflection reflect;
    Specular specular;
  };
  int which;

  __host__ __device__ BxDFVariant() : which(-1) {}
  BxDFVariant(const Lambertian &l) : lambert(l), which(1) {}
  BxDFVariant(const OrenNayar &o) : oren(o), which(2) {}
  BxDFVariant(const AreaLight &li) : light(li), which(3) {}
  BxDFVariant(const TorranceSparrow &ts) : microfacet(ts), which(4) {}
  BxDFVariant(const SpecularReflection &r) : reflect(r), which(5) {}
  BxDFVariant(const Specular &spec) : specular(spec), which(6) {}
  __host__ __device__ BxDFVariant(const BxDFVariant &b);

  __host__ __device__ BxDFVariant& operator=(const BxDFVariant &b);

  __device__ Vec3 f(const Vec3 &wo, const Vec3 &wi) const;
  __device__ Vec3 sample_f(const Vec3 &wo, Vec3 *wi, float u, float v, int face, float *pdf) const;
  __device__ float pdf(const Vec3 &wo, const Vec3 &wi) const;
  __host__ __device__ bool is_light() const;
  __device__ Vec3 emittance() const;
  __device__ void tex_update(const Vector<Texture> &tex_arr, float u, float v);
};
