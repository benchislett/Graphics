#include "bvh.cuh"

#include <cstdint>
#include <vector>
#include <algorithm>
#include <tuple>
#include <cstdio>
#include <cstdio>

bool compareX(Primitive &a, Primitive &b) {
  return a.bound.lower.x < b.bound.lower.x;
}

bool compareY(Primitive &a, Primitive &b) {
  return a.bound.lower.y < b.bound.lower.y;
}

bool compareZ(Primitive &a, Primitive &b) {
  return a.bound.lower.z < b.bound.lower.z;
}


BVHNode::BVHNode(int count, Primitive *primitives, BVHNode *nodes, int start, int end, int idx) {
  int axis = rand() % 3;
  auto comparator = (axis == 0) ? compareX : (axis == 1) ? compareY : compareZ;

  int span = end - start;

  if (span == 1) {
    left = -start - 1;
    leftBound = primitives[start].bound;
    right = left;
    rightBound = leftBound;
  } else if (span == 2) {
    left = -start - 1;
    leftBound = primitives[start].bound;
    right = -start - 2;
    rightBound = primitives[start + 1].bound;
  } else {
    std::sort(primitives + start, primitives + end, comparator);
    int midpoint = start + span / 2;
    left = 2 * idx + 1;
    right = 2 * idx + 2;
    BVHNode *leftNode = new BVHNode(count, primitives, nodes, start, midpoint, left);
    BVHNode *rightNode = new BVHNode(count, primitives, nodes, midpoint, end, right);
    leftBound = (AABB){fminf(leftNode->leftBound.lower, leftNode->rightBound.lower), fmaxf(leftNode->leftBound.upper, leftNode->rightBound.upper)};
    rightBound = (AABB){fminf(rightNode->leftBound.lower, rightNode->rightBound.lower), fmaxf(rightNode->leftBound.upper, rightNode->rightBound.upper)};
  }

  nodes[idx] = *this;
}

BVH::BVH(int count, Primitive *primitives) {
  printf("Building bvh...\n");
  if (count == 0) {
    nodeCount = 0;
    nodes = NULL;
    return;
  }

  nodeCount = count - 1;
  nodes = (BVHNode *)malloc(nodeCount * sizeof(BVHNode));

  auto root = new BVHNode(count, primitives, nodes, 0, count, 0);

  for (int i = 0; i < nodeCount; i++) {
    printf("Node %d with children %d and %d\n", i, nodes[i].left, nodes[i].right);   
  }

  printf("BVH built\n");
}

