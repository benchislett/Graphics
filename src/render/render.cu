#include "camera.cuh"
#include "image.cuh"
#include "render.cuh"
#include "triangle.cuh"

#include <cuda.h>
#include <iostream>

#define cudaCheckError()                                                               \
  {                                                                                    \
    cudaError_t e = cudaGetLastError();                                                \
    if (e != cudaSuccess) {                                                            \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(0);                                                                         \
    }                                                                                  \
  }

__global__ void render_kernel_normals(Triangle s, Camera cam, float3* out, unsigned int w, unsigned int h) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x >= w || y >= h)
    return;

  TriangleNormals normals(s);

  int spp_x  = 64;
  int spp_y  = 64;
  int spp    = spp_x * spp_y;
  float3 rgb = {0, 0, 0};

  for (int xx = 0; xx < spp_x; xx++) {
    for (int yy = 0; yy < spp_y; yy++) {
      float u = ((float) x + (float) xx / (float) spp_x) / (float) w;
      float v = ((float) y + (float) yy / (float) spp_y) / (float) h;

      Ray r = cam.get_ray(u, v);


      auto i = s.intersects(r);
      if (i.hit) {
        auto normal = normalized(normals.at(i.uvw, r));
        rgb.x += (normal.x + 1.0) / 2.0;
        rgb.y += (normal.y + 1.0) / 2.0;
        rgb.z += (normal.z + 1.0) / 2.0;
      }
    }
  }

  out[y * w + x] = rgb / (float) spp;
}

Image render_normals(Triangle s, Camera cam, unsigned int w, unsigned int h) {
  Image out(w, h);

  float3* device_out;
  cudaMalloc(&device_out, w * h * sizeof(float3));
  cudaCheckError();

  dim3 block(32, 32);
  dim3 grid((w + 31) / 32, (h + 31) / 32);
  render_kernel_normals<<<grid, block>>>(s, cam, device_out, w, h);
  cudaCheckError();

  cudaDeviceSynchronize();
  cudaMemcpy(out.data, device_out, w * h * sizeof(float3), cudaMemcpyDeviceToHost);
  cudaCheckError();
  cudaFree(device_out);

  return out;
}
