#pragma once

#include "math.cuh"
#include "vector.cuh"
#include "camera.cuh"
#include "bvh.cuh"
#include "random.cuh"
#include "texture.cuh"

struct Scene {
  Camera cam;
  BVH b;
  Vector<Primitive> prims;
  Vector<int> lights;
  Vector<BSDF> materials;
  Vector<Texture> textures;
  RNG gen;
};
