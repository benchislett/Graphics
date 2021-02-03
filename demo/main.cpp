#include "camera.h"
#include "cu_math.h"
#include "geometry.h"
#include "integrate.h"
#include "scene.h"

// clang-format off
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>
// clang-format on

float3 position       = make_float3(0.f);
float3 look_direction = make_float3(1.f, 0.f, 0.f);

void init_gl(int* argc, char** argv) {
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(256, 256);
  glutCreateWindow("Test render window");
  glewInit();
}

void exit_gl() {}

int main(int argc, char** argv) {
  init_gl(&argc, argv);

  float3 position = make_float3(0.f, 0.f, -10.f);
  float3 target   = make_float3(0.f, 0.f, 2.f);
  Scene scene     = from_obj("data/test.obj");

  Camera camera = make_camera(position, target, 1.57f, 1.f);
  Image image   = render(camera, scene, 256, 256, 10);
  to_ppm(image, "output.ppm");

  return 0;
}
