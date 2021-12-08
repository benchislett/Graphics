#pragma once

#include "../camera/camera.cuh"
#include "../geometry/trimesh.cuh"
#include "../image/image.cuh"

Image render_normals(Triangle tri, Camera cam, unsigned int w, unsigned int h);
