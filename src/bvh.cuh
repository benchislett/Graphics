#pragma once

#include "primitive.cuh"

Slab bounding_slab(const Slab &s);
Slab bounding_slab(const Slab &s1, const Slab &s2);
Slab bounding_slab(const Tri &t);

struct BVHNode {
  Slab s;
  int left;
  int right;
}

struct BVH {
  int n_nodes;
  int n_tris;
  BVHNode *nodes;
  Tri *tris;
}

BVH build_bvh(Tri *tris, int n);
