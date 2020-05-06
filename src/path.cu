#include "path.cuh"
#include "intersection.cuh"
#include "random.cuh"

__device__ Vec3 sample_li(const Intersection &i, const Primitive &prim, const Scene &s, Vec3 *wi, float u, float v, float *pdf, bool *vis) {
  Vec3 light_p;
  Vec3 light_n;
  prim.t.sample(u, v, pdf, &light_p, &light_n);

  if (*pdf == 0.f || light_p == i.p) {
    *pdf = 0.f;
    *vis = false;
    return Vec3(0.f, 0.f, 0.f);
  }
  *wi = normalized(light_p - i.p);
  *pdf *= length_sq(light_p - i.p) / dot_abs(*wi, light_n);
  *vis = hit_first(Ray(i.p, (*wi)), s, &prim);

  // record visibility
  return s.materials[prim.bsdf].emittance();
}

__device__ float power_heuristic(float nf, float f_pdf, float ng, float g_pdf) {
  float f = nf * f_pdf;
  float g = ng * g_pdf;
  return (f * f) / (f * f + g * g);
}

__device__ Vec3 direct_lighting(const Intersection &i, const Primitive &light, const Scene &s, float u_scatter, float v_scatter, float u_light, float v_light, int bxdf_choice) {
  Vec3 ld = {0.f, 0.f, 0.f};
  Vec3 wi;
  float light_pdf = 0.f, scatter_pdf = 0.f;
  bool visible;

  BSDF& mat = s.materials[i.prim->bsdf];

  Vec3 f;
  float weight;

  // Sample light
  Vec3 li = sample_li(i, light, s, &wi, u_light, v_light, &light_pdf, &visible);
  if (light_pdf != 0.f && !is_zero(li)) {
    f = mat.f(i.incoming, wi) * dot_abs(wi, i.n);
    scatter_pdf = mat.pdf(i.incoming, wi);

    if (!is_zero(f)) {
      if (!visible) li = {0.f, 0.f, 0.f};

      if (!is_zero(li)) {
        weight = power_heuristic(1.f, light_pdf, 1.f, scatter_pdf);
        ld += f * li * weight / light_pdf;
      }
    }
  }

  // Sample BSDF
  /*
  f = mat.sample_f(i.incoming, &wi, u_scatter, v_scatter, &scatter_pdf, bxdf_choice);
  f *= dot_abs(wi, i.n);
  if (!is_zero(f) && scatter_pdf != 0.f) {
    weight = power_heuristic(1.f, scatter_pdf, 1.f, light_pdf);
    bool did_hit = hit_first(Ray(i.p, wi), s, &light);
    if (did_hit) ld += f * li * weight / scatter_pdf;
  }*/

  return ld;
}

__device__ Vec3 sample_one_light(const Intersection &i, const Scene &s, float u_scatter, float v_scatter, float u_light, float v_light, int light_idx, int bxdf_idx) {
  if (s.lights.size() == 0) return Vec3(0.f);

  return direct_lighting(i, s.prims[s.lights[light_idx]], s, u_scatter, v_scatter, u_light, v_light, bxdf_idx) * (float)s.lights.size();
}

__device__ Vec3 trace(const Ray &r, const Scene &scene, LocalDeviceRNG &gen, int max_depth) {
  Vec3 l = {0.f, 0.f, 0.f};
  Vec3 beta = {1.f, 1.f, 1.f};
  Ray ray = r;
  Vec3 wo_world, wi_world;
  float pdf;
  Vec3 f;
  Vec3 uvw;

  bool does_hit;
  Intersection i;
  int bounces;
  for (bounces = 0;; bounces++) {
    does_hit = hit(ray, scene, &i);

    if (!does_hit || bounces >= max_depth) {
      // l = beta * Vec3(1.f, 1.f, 1.f);
      break;
    }

    BSDF &mat = scene.materials[i.prim->bsdf];

    uvw = {i.u, i.v, 1.f - i.u - i.v};
    uvw = (i.prim->t.t_a * uvw.e[2]) + (i.prim->t.t_b * uvw.e[0]) + (i.prim->t.t_c * uvw.e[1]);
    mat.update(i.n, i.s, scene.textures, uvw.e[0], uvw.e[1]);
    
    if (mat.is_light() && bounces == 0) {
      l += beta * mat.emittance();
      break;
    }

    int n = mat.n_bxdfs;
    int light_choice = gen.generate_int(0, scene.lights.size() - 1);
    int bxdf_choice = (n == 1) ? 0 : gen.generate_int(0, n - 1);
    l += beta * sample_one_light(i, scene, gen.generate(), gen.generate(), gen.generate(), gen.generate(), light_choice, bxdf_choice);

    wo_world = i.incoming;
    int choice = (n == 1) ? 0 : gen.generate_int(0, n - 1);
    f = mat.sample_f(wo_world, &wi_world, gen.generate(), gen.generate(), &pdf, choice);

    if (is_zero(f) || fabsf(pdf) < 0.0001f) break;

    float cos_term = dot_abs(wi_world, i.n);
    beta *= f * cos_term / pdf;
    ray = Ray(i.p, wi_world);
  }
  return l;
}
