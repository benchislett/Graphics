#include "render.cuh"
#include "intersection.cuh"
#include "path.cuh"

#include <random>
#include <functional>

void Render(const RenderParams &params, const Scene &scene, Image &im) {
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
      for (c = 0; c < params.spp; c++) {
        u = ((float)i + rand()) / (float)im.width;
        v = ((float)j + rand()) / (float)im.height;
        colour += trace(scene.cam.get_ray(u, v), scene, 5);
      }
      colour /= (float)params.spp;
      im.film[idx] = colour;
    }
  }
}
