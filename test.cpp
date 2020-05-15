#include "src/render.cuh"
#include "src/io.cuh"
#include "src/intersection.cuh"
#include "src/bvh.cuh"
#include "src/random.cuh"

int main() {
  int width = 1024;
  int height = 1024;
  int spp = 1;

  float vfov = 0.79f;
  float aspect = 1.f;

  //Vec3 look_from(5.03f, 0.91f, -2.20f);
  //Vec3 look_at(-0.21f, 0.83f, -0.34f);
  Vec3 look_from(0.f, 1.f, 3.5f);
  Vec3 look_at(0.f, 1.f, -1.f);

  Vec3 view_up(0.f, 1.f, 0.f);

  Scene s;
  load_obj("data/cornell.obj", &s);
  s.cam = Camera(vfov, aspect, look_from, look_at, view_up);

  RenderParams params = {spp};

  Image im(width, height);

  render(params, s, im);
  
  write_ppm("cornell.ppm", im);
  
  return 0;
}
