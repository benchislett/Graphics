#pragma once

#include "math.cuh"
#include "cuda.cuh"
#include "ray.cuh"
#include "primitive.cuh"
#include "bvh.cuh"
#include "scene.cuh"
 
struct Intersection {
  float t;
  Vec3 p;
  Vec3 n;
  Vec3 s;
  Vec3 incoming;
  Primitive prim;
  float u;
  float v;
  int face;
};

__device__ inline bool hit_test(const Ray &r, const Slab &s);
__device__ bool hit_first(const Ray &r, const Scene &s, const Primitive &p);

__device__ bool hit(const Ray &r, Primitive p, Intersection *i);
__device__ bool hit(const Ray &r, const Scene &s, Intersection *i);
