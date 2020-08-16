#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "camera.cuh"
#include "scene.cuh"

struct Image {
  int width;
  int height;
  float3 *data;
};

void render(const Camera &camera, Scene &scene, Image &image);
