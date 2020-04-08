#pragma once

#include "primitive.cuh"

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
