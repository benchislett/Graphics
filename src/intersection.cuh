#pragma once

#include "math.cuh"
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
  Primitive *prim;
  float u;
  float v;
};

bool hit_test(const Ray &r, const Slab &s);
bool hit_first(const Ray &r, const Scene &s, const Primitive *p);

bool hit(const Ray &r, Primitive &p, Intersection *i);
bool hit(const Ray &r, const Scene &s, Intersection *i);
