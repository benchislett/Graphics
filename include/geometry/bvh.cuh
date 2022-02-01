#pragma once

#include "aabb.cuh"
#include "tri_array.cuh"

struct BVHNode {
  AABB box;
  int right;
  int left;
};

struct BVH {
  TriangleArray primitives;
  Vector<BVHNode> tree;

  __host__ BVH(TriangleArray tris);

  __host__ __device__ TriangleArrayIntersection intersects(Ray r) const;
};
