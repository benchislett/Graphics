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

  cudaChannelFormatDesc channelDesc;
  cudaArray_t cuArray;
  struct cudaResourceDesc resDesc;
  struct cudaTextureDesc texDesc;
  cudaTextureObject_t texObj;

  texture<float4, 1> t_;

  __host__ BVH(TriangleArray tris);

  __device__ BVHNode fetch_node(int idx) const;

  __device__ TriangleArrayIntersection intersects(Ray r) const;
};
