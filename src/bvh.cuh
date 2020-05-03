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

  void to_host() { nodes.to_host(); }
  void to_device() { nodes.to_device(); }
};

BVH build_bvh(const Vector<Primitive> &prims);
