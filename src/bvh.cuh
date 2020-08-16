#pragma once

#include "primitive.cuh"

struct BVHNode {
  int left;
  AABB leftBound;

  int right;
  AABB rightBound;

  BVHNode(int count, Primitive *primitives, BVHNode *nodes, int start, int end, int idx);
};

struct BVH {
  int nodeCount;
  BVHNode *nodes;

  AABB rootBounds;

  BVH(int count, Primitive *primitives);
};

