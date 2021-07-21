#pragma once

#include "../camera/camera.cuh"
#include "../geometry/sphere.cuh"
#include "../image/image.cuh"

Image render_normals(Sphere s, Camera cam, unsigned int w, unsigned int h);
