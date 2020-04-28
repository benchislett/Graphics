#pragma once

#include "math.cuh"
#include "camera.cuh"
#include "bvh.cuh"
#include "random.cuh"
#include "texture.cuh"

struct Scene {
  Camera cam;
  BVH b;
  Primitive **lights;
  int n_lights;
  BSDF *materials;
  int n_materials;
  RNG gen;
  Texture *textures;
  int n_textures;
};
