#include "path.cuh"
#include "intersection.cuh"

#include <random>
#include <functional>

Vec3 sample_li(const Intersection &i, const Primitive &prim, const Scene &s, Vec3 *wi, float u, float v, float *pdf, bool *vis) {
  Vec3 p = prim.t.sample(u, v, pdf);

  if (*pdf == 0.f || p == i.p) {
    *pdf = 0.f;
    *vis = false;
    return Vec3(0.f, 0.f, 0.f);
  }
  *wi = normalized(p - i.p);
  *vis = hit_first(Ray(i.p, (*wi)), s.b, &prim);

  // record visibility
  return prim.bsdf->emittance();
}

float power_heuristic(float nf, float f_pdf, float ng, float g_pdf) {
  float f = nf * f_pdf;
  float g = ng * g_pdf;
  return (f * f) / (f * f + g * g);
}

Vec3 direct_lighting(const Intersection &i, const Primitive &light, const Scene &s, float u_scatter, float v_scatter, float u_light, float v_light) {
  Vec3 ld = {0.f, 0.f, 0.f};
  Vec3 wi;
  float light_pdf = 0.f, scatter_pdf = 0.f;
  bool visible;
  Vec3 li = sample_li(i, light, s, &wi, u_light, v_light, &light_pdf, &visible);

  Vec3 f;
  float weight;
  if (light_pdf != 0.f && !is_zero(li)) {
    f = i.prim->bsdf->f(i.incoming, wi) * dot_abs(wi, i.n);
    scatter_pdf = i.prim->bsdf->pdf(i.incoming, wi);

    if (!is_zero(f)) {
      if (!visible) li = {0.f, 0.f, 0.f};

      if (!is_zero(li)) {
        weight = power_heuristic(1.f, light_pdf, 1.f, scatter_pdf);
        ld += f * li * weight / light_pdf;
      }
    }
  }

  // TODO: Multiple importance sampling
  return ld;
}

Vec3 sample_one_light(const Intersection &i, const Scene &s, float u_scatter, float v_scatter, float u_light, float v_light, float u_choice) {
  if (s.n_lights == 0) return Vec3(0.f, 0.f, 0.f);

  int light_idx = (int)(v_light * s.n_lights);
  return direct_lighting(i, *s.lights[light_idx], s, u_scatter, v_scatter, u_light, v_light) * (float)s.n_lights;
}

Vec3 trace(const Ray &r, const Scene &scene, int max_depth) {
  Vec3 l = {0.f, 0.f, 0.f};
  Vec3 beta = {1.f, 1.f, 1.f};
  bool specular_bounce = false;
  Ray ray = r;
  Vec3 wo_world, wi_world;
  float pdf, u, v;
  Vec3 f;

  std::default_random_engine generator(std::random_device{}());
  std::uniform_real_distribution<float> distribution(0.f, 1.f);
  auto rand = std::bind(distribution, generator);

  bool does_hit;
  Intersection i;
  int bounces;
  for (bounces = 0;; bounces++) {
    does_hit = hit(ray, scene.b, &i);

    if (!does_hit || specular_bounce) {
      // l += beta * Vec3(0.1f, 0.1, 0.75f);
    }

    if (!does_hit || bounces >= max_depth) break;

    i.prim->bsdf->update(i.n, i.s);
    
    if (i.prim->bsdf->is_light() && bounces == 0) {
      l += beta * i.prim->bsdf->emittance();
      break;
    }

    l += beta * sample_one_light(i, scene, rand(), rand(), rand(), rand(), rand());

    // Sample illumination
    // When area lights are implemented

    wo_world = i.incoming;
    u = rand();
    v = rand();
    f = i.prim->bsdf->sample_f(wo_world, &wi_world, u, v, &pdf);
    if (is_zero(f) || pdf == 0.f) break;

    float cos_term = dot_abs(wi_world, i.n);
    beta *= f * cos_term / pdf;
    ray = Ray(i.p, wi_world);
  }
  return l;
}
