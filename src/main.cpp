#include "cu_math.h"
#include "geometry.h"

#include <iostream>

int main() {
  Ray r;
  r.origin    = make_float3(0.f);
  r.direction = make_float3(1.f, 0.f, 0.f);

  Triangle t;
  t.v0 = make_float3(5.f, 2.f, 0.f);
  t.v1 = make_float3(5.f, -1.f, 1.f);
  t.v2 = make_float3(5.f, -1.f, -1.f);

  TriangleHitRecord rec = hit(r, t);

  std::cout << rec.hit << '\n';
  std::cout << rec.time << '\n';

  AABB b;
  b.lo = make_float3(5.f, -2.f, -2.f);
  b.hi = make_float3(5.f, 2.f, 2.f);

  HitRecord recBox = hit(r, b);

  std::cout << recBox.hit << '\n';
  std::cout << recBox.time << '\n';

  return 0;
}
