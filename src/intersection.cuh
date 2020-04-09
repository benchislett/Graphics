#pragma once

#include "math.cuh"
#include "ray.cuh"
#include "primitive.cuh"

struct Intersection {
  float t;
  Vec3 p;
  Vec3 n;
};

bool hit(const Ray &r, const Tri &t, Intersection *i);
bool hit(const Ray &r, Tri *tris, int n, Intersection *i);

bool hit_test(const Ray &r, const Slab &s);
