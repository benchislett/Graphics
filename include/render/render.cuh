#pragma once

#include "../camera/camera.cuh"
#include "../geometry/trimesh.cuh"
#include "../image/image.cuh"

Image render_normals(TriMesh host_mesh, Camera cam, unsigned int w, unsigned int h);
