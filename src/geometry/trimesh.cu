#include "float.cuh"
#include "float3.cuh"
#include "trimesh.cuh"

#include <cassert>
#include <cstdio>


__host__ __device__ TriMeshIntersection TriMesh::intersects(Ray r) const {
  TriMeshIntersection isect = {};

  for (int i = 0; i < n; i++) {
    auto tri = tris[i];
    auto ii  = tri.intersects(r);
    if (ii.hit && ((!isect.hit) || (ii.time < isect.time))) {
      isect.point = ii.point;
      isect.uvw   = ii.uvw;
      isect.time  = ii.time;
      isect.hit   = true;
      isect.tri   = tri;
    }
  }

  return isect;
}
