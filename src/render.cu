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
  Vec3 colour_tmp;
  
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.f, 1.f);

  auto rand = std::bind(distribution, generator);

  float u, v;
  float progress_interval = 0.1f;
  float progress = 0.f;
  float r;
  for (j = 0; j < im.height; j++) {
    r = (float)j / (float)(im.height);
    if (r >= progress) {
      progress += progress_interval;
      printf("Rendering %d%% complete\n", (int)(100.f * r));
    }
    for (i = 0; i < im.width; i++) {
      idx = j * im.width + i;
      colour = {0.f, 0.f, 0.f};
      for (c = 0; c < params.spp; c++) {
        u = ((float)i + rand()) / (float)im.width;
        v = ((float)j + rand()) / (float)im.height;
        colour_tmp = trace(scene.cam.get_ray(u, v), scene, 50);
        colour_tmp.e[0] = fmax(fmin(colour_tmp.e[0], 1.f), 0.f);
        colour_tmp.e[1] = fmax(fmin(colour_tmp.e[1], 1.f), 0.f);
        colour_tmp.e[2] = fmax(fmin(colour_tmp.e[2], 1.f), 0.f);
        colour += colour_tmp;
      }
      colour /= (float)params.spp;
      im.film[idx] = colour;
    }
  }
}
