#include "src/render.cuh"
#include "src/io.cuh"
#include "src/intersection.cuh"

int main() {

  int width = 64;
  int height = 64;
  int spp = 1;

  float vfov = 0.698132;
  float aspect = 1; // square

  Vec3 look_from = {1.f, 1.f, 1.f};
  Vec3 look_at = {0.f, 0.f, 0.f};
  Vec3 view_up = {0.f, 1.f, 0.f};

  Camera c(vfov, aspect, look_from, look_at, view_up);

  Vec3 background = {0.2f, 0.2f, 0.75f};

  Scene s = {c, NULL, 0, background, spp};
  load_tris_obj("data/bunny.obj", &s);

  Image im = {width, height, NULL};
  im.film = (Vec3 *)malloc(width * height * sizeof(Vec3));
  
  Render(s, im);
  
  write_tris_ppm("bunny.ppm", im);
  
  return 0;
}
