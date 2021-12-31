#include "float.cuh"
#include "float3.cuh"
#include "tri_array.cuh"

#include <cassert>
#include <cstdio>


__host__ __device__ TriangleArrayIntersection TriangleArray::intersects(Ray r) const {
  TriangleArrayIntersection isect = {};

  for (size_t i = 0; i < size; i++) {
    auto tri = data[i];
    auto ii  = tri.intersects(r);
    if (ii.hit && ((!isect.hit) || (ii.time < isect.time))) {
      isect.point = ii.point;
      isect.uvw   = ii.uvw;
      isect.time  = ii.time;
      isect.hit   = true;
      isect.tri   = tri;
      isect.idx   = i;
    }
  }

  return isect;
}
