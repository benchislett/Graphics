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
#include <string>
// clang-format on

constexpr int width    = 256;
constexpr int height   = 256;
constexpr float aspect = (float) width / (float) height;

// constexpr const std::string model_file = "data/test.obj";

float3 position       = make_float3(0.f);
float3 look_direction = make_float3(1.f, 0.f, 0.f);

GLuint pixel_buffer_obj = 0;
GLuint texture_object   = 0;
struct cudaGraphicsResource* cuda_buffer_resource;

void init_gl(int* argc, char** argv) {
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(256, 256);
  glutCreateWindow("Test render window");
  glewInit();
}

void init_buffer() {
  glGenBuffers(1, &pixel_buffer_obj);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixel_buffer_obj);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * width * height * sizeof(GLubyte), 0, GL_STREAM_DRAW);
  glGenTextures(1, &texture_object);
  glBindTexture(GL_TEXTURE_2D, texture_object);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  cudaGraphicsGLRegisterBuffer(&cuda_buffer_resource, pixel_buffer_obj, cudaGraphicsMapFlagsWriteDiscard);
}

void draw_texture() {
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glEnable(GL_TEXTURE_2D);
  glBegin(GL_QUADS);

  glTexCoord2f(0.0f, 0.0f);
  glVertex2f(0, 0);

  glTexCoord2f(0.0f, 1.0f);
  glVertex2f(0, height);

  glTexCoord2f(1.0f, 0.0f);
  glVertex2f(width, 0);

  glTexCoord2f(1.0f, 1.0f);
  glVertex2f(width, height);

  glEnd();
  glDisable(GL_TEXTURE_2D);
}

void draw() {
  uchar4* d_out;
  cudaGraphicsMapResources(1, &cuda_buffer_resource, 0);
  cudaGraphicsResourceGetMappedPointer((void**) &d_out, NULL, cuda_buffer_resource);

  // DO STUFF WITH D_OUT;

  cudaGraphicsUnmapResources(1, &cuda_buffer_resource, 0);
}

void display() {
  draw();
  draw_texture();
  glutSwapBuffers();
}

void exit_gl() {
  if (pixel_buffer_obj) {
    cudaGraphicsUnregisterResource(cuda_buffer_resource);
    glDeleteBuffers(1, &pixel_buffer_obj);
    glDeleteTextures(1, &texture_object);
  }
}

int main(int argc, char** argv) {
  /*Scene scene   = from_obj(model_file);
  Camera camera = make_camera(position, look_direction - position, 1.57f, aspect);

  Image image = render(camera, scene, 256, 256, 10);
  to_ppm(image, "output.ppm");*/

  init_gl(&argc, argv);
  gluOrtho2D(0, width, height, 0);
  glutDisplayFunc(display);
  init_buffer();
  glutMainLoop();
  atexit(exit_gl);

  return 0;
}

