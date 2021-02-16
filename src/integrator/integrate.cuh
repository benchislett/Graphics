#include "camera.cuh"
#include "cu_math.cuh"
#include "image.cuh"
#include "scene.cuh"

Image render(const Camera camera, const Scene scene, int x, int y, int spp);
