#include "bvh.cuh"
#include "host_rand.cuh"
#include "scoped_timer.cuh"
#include "triangle.cuh"

#include <algorithm>
#include <iostream>
#include <queue>
#include <stack>
#include <tuple>
#include <vector>

__host__ BVH::BVH(TriangleArray tris) : primitives(tris) {
  ScopedMicroTimer x_([&](int us) { printf("BVH constructed in %.2f ms\n", (double) us / 1000.0); });

  auto sort_range = [&](int low, int high, int axis) {
    if (high - low < 2) {
      return;
    }

    auto comp = [&](int a, int b) {
      auto t1 = primitives.tris[a];
      auto t2 = primitives.tris[b];
      if (axis == 0) {
        return (t1.v0.x + t1.v1.x + t1.v2.x) < (t2.v0.x + t2.v1.x + t2.v2.x);
      } else if (axis == 1) {
        return (t1.v0.y + t1.v1.y + t1.v2.y) < (t2.v0.y + t2.v1.y + t2.v2.y);
      } else {
        return (t1.v0.z + t1.v1.z + t1.v2.z) < (t2.v0.z + t2.v1.z + t2.v2.z);
      }
    };

    std::vector<int> idxs(high - low);
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), comp);

    for (int i = 0; i < high - low - 1; i++) {
      int idx = idxs[i];
      while (idx < i) {
        idx = idxs[idx];
      }

      std::swap(primitives.tris[low + idx], primitives.tris[low + i]);
      std::swap(primitives.tri_normals[low + idx], primitives.tri_normals[low + i]);
    }
  };

  std::queue<std::pair<int, int>> tq;
  tq.push(std::make_pair(0, primitives.tris.size));

  // std::random_shuffle(primitives.tris.begin(), primitives.tris.end());
  sort_range(0, primitives.tris.size, rand_in_range(0, 2));

  while (!tq.empty()) {
    auto [low, high] = tq.front();
    tq.pop();

    if (high - low == 1) {
      tree.push_back(BVHNode{AABB(primitives.tris[low]), -low - 1, -low - 1});
    } else {
      sort_range(low, high, rand_in_range(0, 2));

      int cur_idx  = tree.size;
      int next_idx = cur_idx + tq.size() + 1;
      tree.push_back(BVHNode{AABB(), next_idx, next_idx + 1});

      int mid = (high + low) / 2;
      tq.push(std::make_pair(low, mid));
      tq.push(std::make_pair(mid, high));
    }
  }

  for (auto it = std::prev(tree.end()); it >= tree.begin(); it--) {
    BVHNode node = *it;
    if (node.left >= 0 && node.right >= 0) {
      BVHNode left  = tree[node.left];
      BVHNode right = tree[node.right];
      it->box       = left.box.plus(right.box);
    }
  }
}

__host__ __device__ TriangleArrayIntersection BVH::intersects(Ray r) const {
  TriangleArrayIntersection isect = {};
  int stack[64];
  int n_stack = 0;

  if (!tree[0].box.intersects(r).hit)
    return isect;

  stack[n_stack++] = 0;

  while (n_stack > 0) {
    BVHNode node = tree[stack[--n_stack]];

    if (node.left < 0 || node.right < 0) {
      auto i = primitives.intersects(r, -(node.left + 1));

      if (i.hit && (!isect.hit || i.time < isect.time))
        isect = i;

    } else {
      auto left  = tree[node.left];
      auto right = tree[node.right];

      auto i_left  = left.box.intersects(r);
      auto i_right = right.box.intersects(r);

      if (i_left.hit)
        stack[n_stack++] = node.left;
      if (i_right.hit)
        stack[n_stack++] = node.right;
    }
  }

  return isect;
}
