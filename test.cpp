#include "src/render.cuh"
#include "src/io.cuh"
#include "src/intersection.cuh"
#include "src/bvh.cuh"

int main() {

  int width = 128;
  int height = 128;
  int spp = 32;

  float vfov = 0.698132f;
  float aspect = 1.f;

  Vec3 look_from = {4.f, 3.f, 4.f};
  Vec3 look_at = {0.f, 0.f, 0.f};
  Vec3 view_up = {0.f, 1.f, 0.f};

  Scene s;
  load_obj("data/car.obj", &s);
  s.cam = Camera(vfov, aspect, look_from, look_at, view_up);

  RenderParams params = {spp};

  Image im = {width, height, NULL};
  im.film = (Vec3 *)malloc(width * height * sizeof(Vec3));

  Render(params, s, im);
  
  write_ppm("car.ppm", im);
  
  return 0;
}
