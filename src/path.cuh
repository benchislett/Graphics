#pragma once

#include "math.cuh"
#include "cuda.cuh"
#include "ray.cuh"
#include "scene.cuh"

__device__ Vec3 trace(const Ray &r, const Scene &scene, LocalDeviceRNG &gen, int max_depth);
