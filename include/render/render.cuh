#pragma once

#include "../camera/camera.cuh"
#include "../geometry/triangle.cuh"
#include "../image/image.cuh"

Image render_normals(Triangle s, Camera cam, unsigned int w, unsigned int h);
