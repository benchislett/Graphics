#pragma once

#include "../camera/camera.cuh"
#include "../geometry/tri_array.cuh"
#include "../geometry/triangle.cuh"
#include "../image/image.cuh"

Image render_normals(TriangleArray tris, Vector<TriangleNormals> normals_arr, Camera cam, unsigned int w,
                     unsigned int h);
