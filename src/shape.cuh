#pragma once

#include "math.cuh"
#include "cuda.cuh"

struct Slab {
  Vec3 ll;
  Vec3 ur;

  Slab();

  Slab(const Vec3 &v1, const Vec3 &v2) : ll(v1), ur(v2) {}

  __host__ __device__ void expand(const Slab &s);
};

__host__ Slab bounding_slab(const Slab &s1, const Slab &s2);
__host__ Slab bounding_slab(const Vec3 &a, const Vec3 &b, const Vec3 &c);

struct Tri {
  Vec3 a;
  Vec3 b;
  Vec3 c;
  Vec3 n_a;
  Vec3 n_b;
  Vec3 n_c;
  Vec3 t_a;
  Vec3 t_b;
  Vec3 t_c;
  Slab bound;

  Tri(const Vec3 &a, const Vec3 &b, const Vec3 &c);
  Tri(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3 &n);
  Tri(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3 &n_a, const Vec3 &n_b, const Vec3 &n_c);
  Tri(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3 &n_a, const Vec3 &n_b, const Vec3 &n_c, const Vec3 &t_a, const Vec3 &t_b, const Vec3 &t_c);

  __host__ __device__ float area() const;
  __device__ Vec3 sample(float u, float v, float *pdf) const;
};
