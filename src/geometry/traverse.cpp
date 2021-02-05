#include "geometry.h"

#include <climits>
#include <cstdio>

HD TriangleHitRecord first_hit(const Ray ray, const Triangle* triangles, int n_triangles, int* which) {
  TriangleHitRecord closest_hit = (TriangleHitRecord){FLT_MAX, 0.f, 0.f, false};
  for (int i = 0; i < n_triangles; i++) {
    TriangleHitRecord result = hit(ray, triangles[i]);
    if (result.hit && result.time < closest_hit.time) {
      closest_hit = result;
      *which      = i;
    }
  }
  return closest_hit;
}
