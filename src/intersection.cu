#include "intersection.cuh"

#define SIGN(a) (((a) > 0) ? 1 : (((a) < 0) ? -1 : 0))

// Moller-Trumbore intersection
__device__ bool hit(const Ray &r, const Primitive &p, Intersection *i) {
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
  i->prim = p;
  i->incoming = -1 * r.d;
  i->u = u;
  i->v = v;

  //Vec3 normal = p.t.n_a;
  float w = 1.f - u - v;
  Vec3 normal = p.t.n_a * w + p.t.n_b * u + p.t.n_c * v;
  int front_face = -SIGN(dot(r.d, normal));
  normal = normalized(normal) * front_face;
  
  i->face = front_face;
  i->n = normal;
  i->s = normalized(cross(normal, edge0));
  return true;
}

__device__ inline bool hit_test(const Ray &r, const Slab &s) {
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

/*
__device__ bool hit(const Ray &r, const Scene &s, BVHNode *current, Intersection *i) {
  if (!hit_test(r, current->s)) {
    return false;
  }

  Intersection i_l, i_r;
  int left = current->left;
  int right = current->right;
  bool hit_left, hit_right;
  if (left < 0) {
    hit_left = hit(r, s.prims[-(left + 1)], &i_l);
  } else {
    hit_left = hit(r, s, (s.b.nodes.data + left), &i_l);
  }

  if (right < 0) {
    hit_right = hit(r, s.prims[-(right + 1)], &i_r);
  } else {
    hit_right = hit(r, s, (s.b.nodes.data + right), &i_r);
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
*/

__device__ bool hit(const Ray &r, const Scene &s, BVHNode *current, Intersection *i) {
  // Iterative traversal with thread-local stack
  int stack[64];
  stack[0] = -1;
  int *stack_current = &stack[1];

  int left, right;
  bool traverse_left, traverse_right;
  bool hit_any = false;
  Intersection i_tmp;
  do {
    left = current->left;
    right = current->right;

    if (left < 0) {
      if (hit(r, s.prims[-1 - left], &i_tmp)) {
        if (!hit_any || i_tmp.t < i->t) {
          *i = i_tmp;
          hit_any = true;
        }
      }
      traverse_left = false;
    } else {
      traverse_left = hit_test(r, s.b.nodes[left].s);
    }

    if (right < 0) {
      if (hit(r, s.prims[-1 - right], &i_tmp)) {
        if (!hit_any || i_tmp.t < i->t) {
          *i = i_tmp;
          hit_any = true;
        }
      }
      traverse_right = false;
    } else {
      traverse_right = hit_test(r, s.b.nodes[right].s);
    }

    if (traverse_left && traverse_right) {
      *stack_current++ = right;
      current = s.b.nodes.data + left;
    } else if (traverse_left) {
      current = s.b.nodes.data + left;
    } else if (traverse_right) {
      current = s.b.nodes.data + right;
    } else {
      left = *--stack_current;
      current = (left == -1) ? NULL : s.b.nodes.data + left;
    }
  } while (current != NULL);

  return hit_any;
}

__device__ bool hit(const Ray &r, const Vector<Primitive> &prims, Intersection *i) {
  bool hit_any = false;
  float t_best = FLT_MAX;
  Intersection tmp;

  bool did_hit;
  for (int c = 0; c < prims.size(); c++) {
    did_hit = hit(r, prims[c], &tmp);
    if (did_hit) hit_any = true;
    if (did_hit && tmp.t < t_best) {
      t_best = tmp.t;
      *i = tmp;
    }
  }
  return hit_any;
}

__device__ bool hit(const Ray &r, const Scene &s, Intersection *i) {
  //bool res = hit(r, s.prims, i);
  bool res = hit(r, s, s.b.nodes.data + s.b.nodes.size() - 1, i);
  return res;
}

__device__ bool hit_first(const Ray &r, const Scene &s, const Primitive &p) {
  Intersection i;
  bool res = hit(r, s, &i);
  if (!res) return false;

  return i.prim == p;
}
