#include "shape.cuh"

Slab::Slab() {
  ll = { FLT_MAX, FLT_MAX, FLT_MAX };
  ur = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
}

void Slab::expand(const Slab &s) {
  ll = min(ll, s.ll);
  ur = max(ur, s.ur);
}

Slab bounding_slab(const Slab &s) {
  return { s.ll - 0.001f, s.ur + 0.001f };
}

Slab bounding_slab(const Slab &s1, const Slab &s2) {
  return { { fmin(s1.ll.e[0], s2.ll.e[0]) - 0.0001f, fmin(s1.ll.e[1], s2.ll.e[1]) - 0.0001f, fmin(s1.ll.e[2], s2.ll.e[2]) - 0.0001f},
           { fmax(s1.ur.e[0], s2.ur.e[0]) + 0.0001f, fmax(s1.ur.e[1], s2.ur.e[1]) + 0.0001f, fmax(s1.ur.e[2], s2.ur.e[2]) + 0.0001f}};
}

Slab bounding_slab(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
  return { { fmin(fmin(a.e[0], b.e[0]), c.e[0]) - 0.0001f,
             fmin(fmin(a.e[1], b.e[1]), c.e[1]) - 0.0001f,
             fmin(fmin(a.e[2], b.e[2]), c.e[2]) - 0.0001f },
           { fmax(fmax(a.e[0], b.e[0]), c.e[0]) + 0.0001f,
             fmax(fmax(a.e[1], b.e[1]), c.e[1]) + 0.0001f,
             fmax(fmax(a.e[2], b.e[2]), c.e[2]) + 0.0001f }};
}

Tri::Tri(const Vec3 &a, const Vec3 &b, const Vec3 &c) : a(a), b(b), c(c), n_a(cross(b - a, c - a)), n_b(n_a), n_c(n_a), t_a(Vec3(-1.f)), t_b(Vec3(-1.f)), t_c(Vec3(-1.f)), bound(bounding_slab(a, b, c)) {}

Tri::Tri(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3 &n) : a(a), b(b), c(c), n_a(n), n_b(n), n_c(n), t_a(Vec3(-1.f)), t_b(Vec3(-1.f)), t_c(Vec3(-1.f)), bound(bounding_slab(a, b, c)) {}

Tri::Tri(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3 &n_a, const Vec3 &n_b, const Vec3 &n_c) : a(a), b(b), c(c), n_a(n_a), n_b(n_b), n_c(n_c), t_a(Vec3(-1.f)), t_b(Vec3(-1.f)), t_c(Vec3(-1.f)), bound(bounding_slab(a, b, c)) {}

Tri::Tri(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3 &n_a, const Vec3 &n_b, const Vec3 &n_c, const Vec3 &t_a, const Vec3 &t_b, const Vec3 &t_c) : a(a), b(b), c(c), n_a(n_a), n_b(n_b), n_c(n_c), t_a(t_a), t_b(t_b), t_c(t_c), bound(bounding_slab(a, b, c)) {}

float Tri::area() const {
  return 0.5 * length(cross(b - a, c - a));
}

Vec3 Tri::sample(float u, float v, float *pdf) const {
  float u_ = sqrtf(u);
  float v_ = u_ * (1.f - v);
  float w_ = u_ * v;
  u_ = 1.f - u_;
  *pdf = 1.f / area();
  return (a * u_) + (b * v_) + (c * w_);
}
