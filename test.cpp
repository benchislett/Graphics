#include "src/render.cuh"
#include "src/io.cuh"
#include "src/intersection.cuh"
#include "src/bvh.cuh"

int main() {

  int width = 512;
  int height = 512;
  int spp = 2048;

  float vfov = 0.698132;
  float aspect = 1; // square

  Vec3 look_from = {0.f, 1.f, 4.f};
  Vec3 look_at = {0.f, 1.f, -1.f};
  Vec3 view_up = {0.f, 1.f, 0.f};

  Scene s = load_obj("data/cornell.obj");
  s.cam = Camera(vfov, aspect, look_from, look_at, view_up);

  RenderParams params = {spp};

  Image im = {width, height, NULL};
  im.film = (Vec3 *)malloc(width * height * sizeof(Vec3));

  Render(params, s, im);
  
  write_ppm("CornellBox.ppm", im);
  
  return 0;
}
