#pragma once

#include "primitive.cuh"
#include "vector.cuh"

struct BVHNode {
  Slab s;
  int left;
  int right;
};

struct BVH {
  Vector<BVHNode> nodes;
};

BVH build_bvh(const Vector<Primitive> &prims);
