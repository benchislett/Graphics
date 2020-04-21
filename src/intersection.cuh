#pragma once

#include "math.cuh"
#include "ray.cuh"
#include "primitive.cuh"
#include "bvh.cuh"

struct Intersection {
  float t;
  Vec3 p;
  Vec3 n;
  Vec3 s;
  Vec3 incoming;
  Primitive *prim;
};

bool hit_test(const Ray &r, const Slab &s);
bool hit_first(const Ray &r, const BVH &b, Primitive *p);

bool hit(const Ray &r, Primitive &p, Intersection *i);
bool hit(const Ray &r, const BVH &b, Intersection *i);
