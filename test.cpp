#include "src/scene.cuh"
#include "src/io.cuh"
#include "src/helper_math.cuh"
#include "src/trace.cuh"

int main() {
  int width = 200;
  int height = 100;

  float vfov = 0.79f;
  float aspect = (float)width / (float)height;

  float3 lookFrom = make_float3(30.f, 7.f, -21.f);
  float3 lookAt = make_float3(29.24f, 6.79f, -20.38f);
  float3 viewUp = make_float3(0.f, 1.f, 0.f);

  Camera c(vfov, aspect, lookFrom, lookAt, viewUp);

  Scene s = loadMesh("data/conference.obj");
  Image image = {width, height, NULL};

  render(c, s, image);

  writePPM("output.ppm", image);

  return 0;
}

