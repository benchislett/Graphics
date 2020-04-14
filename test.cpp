#include "src/render.cuh"
#include "src/io.cuh"
#include "src/intersection.cuh"
#include "src/bvh.cuh"

int main() {

  int width = 32;
  int height = 32;
  int spp = 1024;

  float vfov = 0.698132;
  float aspect = 1; // square

  Vec3 look_from = {1.f, 1.f, 1.f};
  Vec3 look_at = {0.f, 0.f, 0.f};
  Vec3 view_up = {0.f, 1.f, 0.f};

  Camera c(vfov, aspect, look_from, look_at, view_up);

  Vec3 background = {0.2f, 0.2f, 0.75f};

  int n_prims;
  Primitive *prims = load_obj("data/bunny.obj", &n_prims);

  BVH b = build_bvh(prims, n_prims);

  RenderParams params = {spp};
  Scene s = {c, b, background};

  Image im = {width, height, NULL};
  im.film = (Vec3 *)malloc(width * height * sizeof(Vec3));

  Render(params, s, im);
  
  write_ppm("bunny.ppm", im);
  
  return 0;
}
