#pragma once

#include "math.cuh"
#include "ray.cuh"
#include "tri.cuh"

struct Intersection {
  float t;
  Vec3 p;
  Vec3 n;
}

bool hit(const Ray &r, const Tri &t, Intersection *i);
