#pragma once

#include "camera.cuh"
#include "image.cuh"
#include "tri_array.cuh"
#include "triangle.cuh"

Image render_normals(TriangleArray primitives, Camera cam, unsigned int w, unsigned int h);
