#include "bvh.cuh"

#include <tuple>
#include <vector>
#include <algorithm>
#include <atomic>

uint32_t expand_bits(uint32_t x) {
  x = (x * 0x00010001u) & 0xFF0000FFu;
  x = (x * 0x00000101u) & 0x0F00F00Fu;
  x = (x * 0x00000011u) & 0xC30C30C3u;
  x = (x * 0x00000005u) & 0x49249249u;
  return x;
}

uint32_t morton_code(const Vec3 &p) {
  uint32_t x = expand_bits((uint32_t)p.e[0]);
  uint32_t y = expand_bits((uint32_t)p.e[1]);
  uint32_t z = expand_bits((uint32_t)p.e[2]);
  return (x << 2) + (y << 1) + z;
}

uint32_t delta(uint32_t a, uint32_t b) {
  return a ^ b;
}

int compare_codes(const std::pair<Tri, uint32_t> &a, const std::pair<Tri, uint32_t> &b) {
  return std::get<1>(a) < std::get<1>(b);
}

int max_diff(std::vector<std::pair<Tri, uint32_t>> &codes, int start, int end) {
  uint32_t largest = 0;
  int largest_idx = start;
  uint32_t d;
  for (int i = start; i < end; i++) {
    d = delta(std::get<1>(codes[i]), std::get<1>(codes[i + 1]));
    if (d >= largest) {
      largest = d;
      largest_idx = i;
    }
  }
  return largest_idx;
}

struct Node {
  Node *left;
  Node *right;
  int tri_left;
  int tri_right;
};

Node *bvh_r(std::vector<std::pair<Tri, uint32_t>> &codes, int start, int end) {
  if (start > end) return NULL;

  Node *root = (Node *)malloc(sizeof(Node));

  int split = max_diff(codes, start, end);
  
  if (split == start) {
    root->left = NULL;
    root->tri_left = start;
  } else {
    root->left = bvh_r(codes, start, split);
  }

  if (split + 1 == end) {
    root->right = NULL;
    root->tri_right = end;
  } else {
    root->right = bvh_r(codes, split + 1, end);
  }

  return root;
}

int populate_nodes(Tri *tris, Node *root, std::vector<BVHNode> &node_vec) {
  int left, right;
  Slab l, r;
  if (root->left == NULL) {
    left = -root->tri_left;
    l = tris[-left].bound;
  } else {
    left = populate_nodes(tris, root->left, node_vec);
    l = node_vec[left].s;
  }

  if (root->right == NULL) {
    right = -root->tri_right;
    r = tris[-right].bound;
  } else {
    right = populate_nodes(tris, root->right, node_vec);
    r = node_vec[right].s;
  }

  node_vec.push_back((BVHNode){bounding_slab(l, r), left, right});
  return node_vec.size() - 1;
}

BVH build_bvh(Tri *tris, int n) {
  int i;

  BVHNode *nodes = (BVHNode *)malloc((n - 1) * sizeof(BVHNode));
  if (nodes == NULL) return {0, n, NULL, tris};

  std::vector<std::pair<Tri, uint32_t>> codes;
  
  Slab world_bound;
  for (i = 0; i < n; i++) world_bound.expand(tris[i].bound);
  
  Vec3 centroid;
  for (i = 0; i < n; i++) {
    centroid = (tris[i].bound.ll + tris[i].bound.ur) / 2.f;
    centroid = (float)(1 << 10) * ((centroid - world_bound.ll) / (world_bound.ur - world_bound.ll));
    codes.push_back(std::make_pair(tris[i], morton_code(centroid)));
  }

  std::sort(codes.begin(), codes.end(), compare_codes);
  for (i = 0; i < n; i++) tris[i] = std::get<0>(codes[i]);

  Node *root = bvh_r(codes, 0, n - 1);

  std::vector<BVHNode> node_vec;
  populate_nodes(tris, root, node_vec);
  for (i = 0; i < n - 1; i++) nodes[i] = node_vec[i];

  return {n - 1, n, nodes, tris};
}

