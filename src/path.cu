#include "path.cuh"
#include "intersection.cuh"
#include "random.cuh"

__device__ Vec3 sample_li(const Intersection &i, const Primitive &light, const Scene &s, Vec3 *wi, float u, float v, float *pdf, bool *vis) {

  Vec3 light_p;
  Vec3 light_n;
  light.t.sample(u, v, pdf, &light_p, &light_n);

  if (*pdf == 0.f || light_p == i.p) {
    *pdf = 0.f;
    *vis = false;
    return Vec3(0.f, 0.f, 0.f);
  }
  *wi = normalized(light_p - i.p);
  *pdf *= length_sq(light_p - i.p) / dot_abs(*wi, light_n);
  *vis = (dot(*wi, i.n) > 0.f) && hit_first(Ray(i.p, (*wi)), s, light);

  // record visibility
  return light.bsdf.emittance();
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

  Vec3 f;
  float weight;

  // Sample light
  Vec3 li = sample_li(i, light, s, &wi, u_light, v_light, &light_pdf, &visible);
  if (light_pdf != 0.f && !is_zero(li)) {
    f = i.prim.bsdf.f(i.incoming, wi) * dot_abs(wi, i.n);
    scatter_pdf = i.prim.bsdf.pdf(i.incoming, wi);

    if (!is_zero(f)) {
      if (visible) {
        weight = power_heuristic(1.f, light_pdf, 1.f, scatter_pdf);
        ld += f * li * weight / light_pdf;
      }
    }
  }

  // Sample BSDF
  /*
  f = i.prim.bsdf.sample_f(i.incoming, &wi, u_scatter, v_scatter, i.face, &scatter_pdf, bxdf_choice);
  f *= dot_abs(wi, i.n);
  if (scatter_pdf != 0.f && !is_zero(f)) {
    Ray r(i.p, wi);
    Intersection i_light;
    bool res = hit(r, s, &i_light);
    if (res && light == i_light.prim) {
      light_pdf = length_sq(i.p - i_light.p) / (dot_abs(i_light.n, wi) * light.t.area());
      if (light_pdf == 0.f) return ld;
      li = light.bsdf.emittance();
      weight = power_heuristic(1.f, scatter_pdf, 1.f, light_pdf);
      ld += f * li * weight / scatter_pdf;
    }
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

  bool specular_bounce = false;
  bool does_hit;
  Intersection i;
  int bounces;
  for (bounces = 0;; bounces++) {
    does_hit = hit(ray, scene, &i);

    if (bounces == 0 || specular_bounce) {
      if (does_hit) l += beta * i.prim.bsdf.emittance();
      if (length_sq(i.prim.bsdf.emittance()) > 0.f && bounces == 0) break;
    }

    if (!does_hit || bounces >= max_depth) break;

    int n = i.prim.bsdf.n_bxdfs;

    uvw = {i.u, i.v, 1.f - i.u - i.v};
    uvw = (i.prim.t.t_a * uvw.e[2]) + (i.prim.t.t_b * uvw.e[0]) + (i.prim.t.t_c * uvw.e[1]);
    i.prim.bsdf.update(i.n, i.s, scene.textures, uvw.e[0], uvw.e[1]);
    specular_bounce = i.prim.bsdf.is_specular();

    int light_choice = gen.generate_int(0, scene.lights.size() - 1);
    int bxdf_choice = (n == 1) ? 0 : gen.generate_int(0, n - 1);
    l += beta * sample_one_light(i, scene, gen.generate(), gen.generate(), gen.generate(), gen.generate(), light_choice, bxdf_choice);

    wo_world = i.incoming;
    int choice = (n == 1) ? 0 : gen.generate_int(0, n - 1);
    f = i.prim.bsdf.sample_f(wo_world, &wi_world, gen.generate(), gen.generate(), i.face, &pdf, choice);

    if (is_zero(f) || fabsf(pdf) < 0.0001f) break;

    float cos_term = dot_abs(wi_world, i.n);
    beta *= f * cos_term / pdf;
    ray = Ray(i.p, wi_world);
  }
  return l;
}
