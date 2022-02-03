#include "bvh.cuh"
#include "host_rand.cuh"
#include "scoped_timer.cuh"
#include "triangle.cuh"

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <queue>
#include <stack>
#include <tuple>
#include <vector>

constexpr int num_bins            = 32;
constexpr int primitives_per_leaf = 8;

static int total_cnt = 0;

// http://www.sci.utah.edu/~wald/Publications/2007/ParallelBVHBuild/fastbuild.pdf
void Build(Vector<BVHNode>& tree, int root_idx, int num_primitives, int primitives_offset,
           const TriangleArray& primitives) {
  assert(num_primitives > 0);
  assert(num_primitives <= primitives.tris.size);
  assert(primitives_offset >= 0);
  assert(primitives_offset < primitives.tris.size);

  total_cnt++;

  int n_left = 0, n_right = 0;

  tree.size = max((int) tree.size, root_idx + 1);
  assert(tree.size < tree.capacity); // requires pre-allocated capacity

  // Bin primitives along spatial partitions of the bounding box of all their centroids
  AABB bin_region;
  for (int i = primitives_offset; i < num_primitives + primitives_offset; i++) {
    AABB tri_bound(primitives.tris[i]);
    Point3 centroid = tri_bound.centroid();
    bin_region      = bin_region.plus(AABB(centroid, centroid));
  }

  // Base case, if tree is small enough make a leaf node
  if (num_primitives <= primitives_per_leaf) {
    tree[root_idx].left  = -primitives_offset - 1;
    tree[root_idx].right = -num_primitives;
    tree[root_idx].box   = AABB();
    for (int i = primitives_offset; i < num_primitives + primitives_offset; i++) {
      tree[root_idx].box = tree[root_idx].box.plus(AABB(primitives.tris[i]));
    }
    return;
  }

  // Bin according to longest axis of binning region
  float which_bin_coeff_x = num_bins * (1 - 0.00001f) / (float) (bin_region.hi.x - bin_region.lo.x);
  float which_bin_coeff_y = num_bins * (1 - 0.00001f) / (float) (bin_region.hi.y - bin_region.lo.y);
  float which_bin_coeff_z = num_bins * (1 - 0.00001f) / (float) (bin_region.hi.z - bin_region.lo.z);
  auto bin_for_tri        = [&](int idx, int axis) {
    Point3 centroid = AABB(primitives.tris[idx]).centroid();
    if (axis == 0) {
      return (int) (which_bin_coeff_x * (centroid.x - bin_region.lo.x));
    } else if (axis == 1) {
      return (int) (which_bin_coeff_y * (centroid.y - bin_region.lo.y));
    } else {
      return (int) (which_bin_coeff_z * (centroid.z - bin_region.lo.z));
    }
  };

  if (which_bin_coeff_x == INFINITY || which_bin_coeff_y == INFINITY || which_bin_coeff_z == INFINITY) {
    n_left               = num_primitives / 2;
    n_right              = num_primitives - n_left;
    tree[root_idx].left  = tree.size;
    tree[root_idx].right = tree.size + 1;
    tree.size += 2;

    Build(tree, tree[root_idx].left, n_left, primitives_offset, primitives);
    Build(tree, tree[root_idx].right, n_right, primitives_offset + n_left, primitives);
    return;
  }

  // // Determine the cost for splitting along each bin border
  float current_cost = num_primitives * bin_region.surface_area();

  int num_splits = num_bins - 1;

  int min_splits[3];
  float sah_bin_split_costs[num_splits][3] = {};

  for (int axis = 0; axis < 3; axis++) {

    // Calculate bounding box and number of tris per bin
    int num_tris_per_bin[num_bins]       = {};
    AABB centroid_bbox_per_bin[num_bins] = {};
    for (int i = primitives_offset; i < num_primitives + primitives_offset; i++) {
      int which = bin_for_tri(i, axis);

      AABB tri_bbox      = AABB(primitives.tris[i]);
      AABB centroid_bbox = AABB(tri_bbox.centroid(), tri_bbox.centroid());

      num_tris_per_bin[which]++;
      centroid_bbox_per_bin[which] = centroid_bbox_per_bin[which].plus(centroid_bbox);
    }

    // First pass: analyze cost for splits from the left
    for (int i = 1; i < num_bins; i++) {
      int num_tris_on_left = 0;
      AABB bbox_from_left;
      for (int j = 0; j < i; j++) {
        num_tris_on_left += num_tris_per_bin[j];
        bbox_from_left = bbox_from_left.plus(centroid_bbox_per_bin[j]);
      }
      sah_bin_split_costs[i - 1][axis] += num_tris_on_left * bbox_from_left.surface_area();
    }

    // Second pass: analyze cost for splits from the right
    for (int i = num_splits; i > 0; i--) {
      int num_tris_on_right = 0;
      AABB bbox_from_right;
      for (int j = num_splits; j >= i; j--) {
        num_tris_on_right += num_tris_per_bin[j];
        bbox_from_right = bbox_from_right.plus(centroid_bbox_per_bin[j]);
      }
      sah_bin_split_costs[i - 1][axis] += num_tris_on_right * bbox_from_right.surface_area();
    }

    // Determine split with the least cost
    int min_idx = 0;

    for (int i = 1; i < num_splits; i++) {
      if (sah_bin_split_costs[i][axis] < sah_bin_split_costs[min_idx][axis]) {
        min_idx = i;
      }
    }

    int min_split    = min_idx + 1;
    min_splits[axis] = min_split;
  }

  // Determine which axis is cheapest to split on
  int min_axis = 0;
  if (sah_bin_split_costs[min_splits[1] - 1][1] < sah_bin_split_costs[min_splits[min_axis] - 1][min_axis])
    min_axis = 1;
  if (sah_bin_split_costs[min_splits[2] - 1][2] < sah_bin_split_costs[min_splits[min_axis] - 1][min_axis])
    min_axis = 2;

  // Partition primitives
  int begin = primitives_offset;
  int end   = begin + num_primitives;
  while (begin < end - 1) {
    int which = bin_for_tri(begin, min_axis);
    if (which >= min_splits[min_axis]) {
      // left triangle belongs on the right
      while (bin_for_tri(end - 1, min_axis) >= min_splits[min_axis]) {
        end--;
      }
      // right triangle now belongs on the left. swap!
      std::swap(primitives.tris[begin], primitives.tris[end - 1]);
      std::swap(primitives.tri_normals[begin], primitives.tri_normals[end - 1]);
    }
    begin++;
  }

  for (int i = primitives_offset; i < num_primitives + primitives_offset; i++) {
    if (bin_for_tri(i, min_axis) < min_splits[min_axis]) {
      n_left++;
    } else {
      n_right++;
    }
  }

  // Base case, if tree is small enough make a leaf node
  if ((n_left <= primitives_per_leaf || n_right <= primitives_per_leaf)
      || sah_bin_split_costs[min_splits[min_axis] - 1][min_axis] == current_cost) {
    n_left  = num_primitives / 2;
    n_right = num_primitives - n_left;
  }

  assert(n_left + n_right == num_primitives);

  tree[root_idx].left  = tree.size;
  tree[root_idx].right = tree.size + 1;
  tree.size += 2;

  Build(tree, tree[root_idx].left, n_left, primitives_offset, primitives);
  Build(tree, tree[root_idx].right, n_right, primitives_offset + n_left, primitives);
}

__host__ BVH::BVH(TriangleArray tris) : primitives(tris) {
  ScopedMicroTimer x_([&](int us) { printf("BVH constructed in %.2f ms\n", (double) us / 1000.0); });

  tree.reserve(8 * primitives.tris.size + 1);

  Build(tree, 0, primitives.tris.size, 0, primitives);

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

  // return primitives.intersects(r);

  if (!tree[0].box.intersects(r).hit)
    return isect;

  stack[n_stack++] = 0;

  int primitive_tests = 0;
  int biggest_node    = 0;

  while (n_stack > 0) {
    BVHNode node = tree[stack[--n_stack]];

    if (node.left < 0 || node.right < 0) {
      int begin          = -(node.left + 1);
      int num_primitives = -(node.right);
      primitive_tests += num_primitives;
      biggest_node = max(biggest_node, num_primitives);
      for (int i = begin; i < begin + num_primitives; i++) {
        auto ii = primitives.intersects(r, i);

        if (ii.hit && (!isect.hit || ii.time < isect.time))
          isect = ii;
      }
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
    assert(n_stack < 64);
  }

  return isect;
}
