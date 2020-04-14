#pragma once

#include "primitive.cuh"

struct BVHNode {
  Slab s;
  int left;
  int right;
};

struct BVH {
  int n_nodes;
  int n_tris;
  BVHNode *nodes;
  Primitive *prims;
};

BVH build_bvh(Primitive *prims, int n);
