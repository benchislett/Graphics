#pragma once

#include "math.cuh"
#include "cuda.cuh"
#include "vector.cuh"
#include "bxdf.cuh"

struct BSDF {
  Vec3 n, s, t;
  BxDFVariant b[2];
  int n_bxdfs;

  __host__ __device__ BSDF() : n_bxdfs(0) {}
  BSDF(BxDFVariant b1) : b {b1, b1}, n_bxdfs(1) {}
  BSDF(BxDFVariant b1, BxDFVariant b2) : b {b1, b2}, n_bxdfs(2) {}
  __host__ __device__ BSDF(const BSDF &bs) : n(bs.n), s(bs.s), t(bs.t), b {bs.b[0], bs.b[1]}, n_bxdfs(bs.n_bxdfs) {}

  __host__ __device__ BSDF& operator=(const BSDF &bs) { n = bs.n; s = bs.s; t = bs.t; b[0] = bs.b[0]; b[1] = bs.b[1]; n_bxdfs = bs.n_bxdfs; return *this; }

  __device__ void update(const Vec3 &n, const Vec3 &s, const Vector<Texture> &tex_arr, float u = -1.f, float v = -1.f);

  __device__ Vec3 world2local(const Vec3 &v) const;
  __device__ Vec3 local2world(const Vec3 &v) const;

  __device__ Vec3 f(const Vec3 &wo_world, const Vec3 &wi_world) const;
  __device__ Vec3 sample_f(const Vec3 &wo_world, Vec3 *wi_world, float u, float v, int face, float *pdf, int choice) const;
  __device__ float pdf(const Vec3 &wo_world, const Vec3 &wi_world) const;
  __host__ __device__ bool is_light() const;
  __device__ Vec3 emittance() const;
};
