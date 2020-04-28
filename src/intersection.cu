#include "intersection.cuh"
#include <cstdio>

#define SIGN(a) (((a) > 0) ? 1 : (((a) < 0) ? -1 : 0))

// Moller-Trumbore intersection
bool hit(const Ray &r, Primitive &p, Intersection *i) {
  Vec3 edge0 = p.t.b - p.t.a;
  Vec3 edge1 = p.t.c - p.t.a;

  Vec3 h = cross(r.d, edge1);

  float det = dot(edge0, h);

  if (fabsf(det) < FLT_MIN) return false;

  float detInv = 1.0f / det;
  Vec3 s = r.o - p.t.a;
  float u = detInv * dot(s, h);

  if (u < 0.0f || u > 1.0f) return false;

  Vec3 q = cross(s, edge0);
  float v = detInv * dot(r.d, q);

  if (v < 0.0f || u + v > 1.0f) return false;

  float time = detInv * dot(edge1, q);

  if (time < 0.01f) return false;

  i->p = r.at(time);
  i->t = time;
  i->prim = &p;
  i->incoming = -1 * r.d;
  i->u = u;
  i->v = v;

  //Vec3 normal = p.t.n_a;
  float w = 1.f - u - v;
  Vec3 normal = p.t.n_a * w + p.t.n_b * u + p.t.n_c * v;
  normal = normalized(normal) * (-SIGN(dot(r.d, normal)));

  i->n = normal;
  i->s = normalized(cross(normal, edge0));
  return true;
}

bool hit_test(const Ray &r, const Slab &s) {
  float xInv = 1.f / r.d.e[0];
  float yInv = 1.f / r.d.e[1];
  float zInv = 1.f / r.d.e[2];

  float t0 = (s.ll.e[0] - r.o.e[0]) * xInv;
  float t1 = (s.ur.e[0] - r.o.e[0]) * xInv;
  float t2 = (s.ll.e[1] - r.o.e[1]) * yInv;
  float t3 = (s.ur.e[1] - r.o.e[1]) * yInv;
  float t4 = (s.ll.e[2] - r.o.e[2]) * zInv;
  float t5 = (s.ur.e[2] - r.o.e[2]) * zInv;

  float tmin = fmax(fmax(fmin(t0, t1), fmin(t2, t3)), fmin(t4, t5));
  float tmax = fmin(fmin(fmax(t0, t1), fmax(t2, t3)), fmax(t4, t5));

  return (0 < tmax) && (tmin < tmax);
}

bool hit(const Ray &r, const BVH &b, BVHNode *current, Intersection *i) {
  if (!hit_test(r, current->s)) {
    return false;
  }

  Intersection i_l, i_r;
  int left = current->left;
  int right = current->right;
  bool hit_left, hit_right;
  if (left < 0) {
    hit_left = hit(r, b.prims[-(left + 1)], &i_l);
  } else {
    hit_left = hit(r, b, (b.nodes + left), &i_l);
  }

  if (right < 0) {
    hit_right = hit(r, b.prims[-(right + 1)], &i_r);
  } else {
    hit_right = hit(r, b, (b.nodes + right), &i_r);
  }

  if (hit_left && hit_right) {
    if (i_l.t < i_r.t) { *i = i_l; }
    else { *i = i_r; }
    return true;
  } else if (hit_left) {
    *i = i_l;
    return true;
  } else if (hit_right) {
    *i = i_r;
    return true;
  }
  return false;
}

bool hit(const Ray &r, Primitive *prims, int n_prims, Intersection *i) {
  bool hit_any = false;
  float t_best = FLT_MAX;
  Intersection tmp;

  bool did_hit;
  for (int c = 0; c < n_prims; c++) {
    did_hit = hit(r, prims[c], &tmp);
    if (did_hit) hit_any = true;
    if (did_hit && tmp.t < t_best) {
      t_best = tmp.t;
      *i = tmp;
    }
  }
  return hit_any;
}

bool hit(const Ray &r, const BVH &b, Intersection *i) {
  // return hit(r, b.prims, b.n_tris, i);
  bool res = hit(r, b, b.nodes + b.n_nodes - 1, i);
  return res;
}

bool hit_first(const Ray &r, const BVH &b, const Primitive *p) {
  Intersection i;
  bool res = hit(r, b, &i);
  if (!res) return false;

  return (i.prim == p);
}
