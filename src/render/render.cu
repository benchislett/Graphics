#include "camera.cuh"
#include "image.cuh"
#include "render.cuh"
#include "scoped_timer.cuh"
#include "sphere.cuh"
#include "tri_array.cuh"

#include <chrono>
#include <cuda.h>
#include <functional>
#include <iostream>
#include <thread>

#define cudaCheckError()                                                               \
  {                                                                                    \
    cudaError_t e = cudaGetLastError();                                                \
    if (e != cudaSuccess) {                                                            \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(0);                                                                         \
    }                                                                                  \
  }

__global__ void render_kernel_normals(TriangleArray tris, Vector<TriangleNormals> normals_arr, Camera cam, float3* out,
                                      unsigned int w, unsigned int h) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x >= w || y >= h)
    return;

  int spp_x  = 1;
  int spp_y  = 1;
  int spp    = spp_x * spp_y;
  float3 rgb = {0, 0, 0};

  for (int xx = 0; xx < spp_x; xx++) {
    for (int yy = 0; yy < spp_y; yy++) {
      float u = ((float) x + (float) xx / (float) spp_x) / (float) w;
      float v = ((float) y + (float) yy / (float) spp_y) / (float) h;

      Ray r = cam.get_ray(u, v);

      auto i = tris.intersects(r);

      if (i.hit) {
        auto normals = normals_arr[i.idx];

        Vec3 normal = normals.at(i.uvw, r);
        rgb.x += (normal.x + 1.0) / 2.0;
        rgb.y += (normal.y + 1.0) / 2.0;
        rgb.z += (normal.z + 1.0) / 2.0;
      }
    }
  }

  out[y * w + x] = rgb / (float) spp;
}

Image render_normals(TriangleArray tris, Vector<TriangleNormals> normals_arr, Camera cam, unsigned int w,
                     unsigned int h) {
  Image out(w, h);

  ScopedMicroTimer x([&](int us) { printf("Rendered in %.2f ms\n", (double) us / 1000.0); });

  dim3 block(16, 16);
  dim3 grid((w + 15) / 16, (h + 15) / 16);
  render_kernel_normals<<<grid, block>>>(tris, normals_arr, cam, out.values.data, w, h);
  cudaDeviceSynchronize();
  cudaCheckError();

  return out;
}
