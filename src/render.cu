#include "render.cuh"
#include "intersection.cuh"
#include <random>

#define MAX_DEPTH 5

Vec3 trace(const Ray &r, const Scene &scene) {
  Intersection i;
  Ray trace_ray = r;
  Vec3 colour = scene.background;
  int depth = 0;
  while (hit(trace_ray, scene.tris, scene.n_tris, &i)) {
    trace_ray = {i.p, i.n};
    colour *= 0.75;
    if (++depth >= MAX_DEPTH) {
      colour = {0.f, 0.f, 0.f};
      break;
    }
  }
  return colour;
}

void Render(const Scene &scene, Image &im) {
  int i,j;
  int c;
  int idx;
  Vec3 colour;
  
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.f, 1.f);

  auto rand = std::bind(distribution, generator);

  float u, v;
  for (j = 0; j < im.height; j++) {
    for (i = 0; i < im.width; i++) {
      idx = j * im.width + i;
      colour = {0.f, 0.f, 0.f};
      for (c = 0; c < scene.spp; c++) {
        u = ((float)i + rand()) / (float)im.width;
        v = ((float)j + rand()) / (float)im.height;
        colour += trace(scene.cam.get_ray(u, v), scene);
      }
      colour /= (float)scene.spp;
      im.film[idx] = colour;
    }
  }
}
