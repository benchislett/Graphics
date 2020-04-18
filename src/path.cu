#include "path.cuh"
#include "intersection.cuh"

#include <random>
#include <functional>

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
    
    if (i.prim->bsdf->is_light()) {
      l += beta * i.prim->bsdf->emittance();
    }

    // Sample illumination
    // When area lights are implemented

    wo_world = -1 * ray.d;
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
