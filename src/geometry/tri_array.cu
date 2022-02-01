#include "float.cuh"
#include "float3.cuh"
#include "tri_array.cuh"

#include <cassert>
#include <cstdio>


__host__ __device__ TriangleArrayIntersection TriangleArray::intersects(Ray r) const {
  TriangleArrayIntersection isect = {};

  for (size_t i = 0; i < tris.size; i++) {
    auto tri = tris[i];
    auto ii  = tri.intersects(r);
    if (ii.hit && ((!isect.hit) || (ii.time < isect.time))) {
      auto normals = tri_normals[i];

      isect.normal = normals.at(ii.uvw, r);
      isect.point  = ii.point;
      isect.uvw    = ii.uvw;
      isect.time   = ii.time;
      isect.hit    = true;
      isect.tri    = tri;
      isect.idx    = i;
    }
  }

  return isect;
}

__host__ __device__ TriangleArrayIntersection TriangleArray::intersects(Ray r, int idx) const {
  TriangleArrayIntersection isect = {};

  auto tri = tris[idx];
  auto ii  = tri.intersects(r);
  if (ii.hit) {
    auto normals = tri_normals[idx];

    isect.normal = normals.at(ii.uvw, r);
    isect.point  = ii.point;
    isect.uvw    = ii.uvw;
    isect.time   = ii.time;
    isect.hit    = true;
    isect.tri    = tri;
    isect.idx    = idx;
  }

  return isect;
}
