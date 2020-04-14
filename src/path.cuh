#pragma once

#include "math.cuh"
#include "ray.cuh"
#include "scene.cuh"

Vec3 trace(const Ray &r, const Scene &scene, int max_depth);
