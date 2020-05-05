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
  DeviceRNG gen;

  void to_host() {
    b.to_host();
    prims.to_host();
    lights.to_host();
    materials.to_host();
    textures.to_host();
    for (int i = 0; i < textures.size(); i++) textures[i].to_host();
  }

  void to_device() {
    b.to_device();
    prims.to_device();
    lights.to_device();
    materials.to_device();
    for (int i = 0; i < textures.size(); i++) textures[i].to_device();
    textures.to_device();
  }
};
