#include "camera.cuh"
#include "image.cuh"
#include "render.cuh"
#include "triangle.cuh"

#include <cuda.h>
#include <iostream>

__global__ void render_kernel_normals(Triangle s, Camera cam, float3* out, unsigned int w, unsigned int h) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  TriangleNormals normals(s);

  float u = (float) x / (float) w;
  float v = (float) y / (float) h;

  Ray r = cam.get_ray(u, v);

  float3 rgb = {0, 0, 0};

  auto i = s.intersects(r);
  if (i.hit) {
    auto normal = normalized(normals.at(i.uvw, r));
    rgb.x       = (normal.x + 1.0) / 2.0;
    rgb.y       = (normal.y + 1.0) / 2.0;
    rgb.z       = (normal.z + 1.0) / 2.0;
  }

  out[y * w + x] = rgb;
}

Image render_normals(Triangle s, Camera cam, unsigned int w, unsigned int h) {
  Image out(w, h);

  float3* device_out;
  cudaMalloc(&device_out, w * h * sizeof(float3));

  dim3 block(32, 32);
  dim3 grid((w + 31) / 32, (h + 31) / 32);
  render_kernel_normals<<<block, grid>>>(s, cam, device_out, w, h);

  cudaDeviceSynchronize();
  cudaMemcpy(out.data, device_out, w * h * sizeof(float3), cudaMemcpyDeviceToHost);
  cudaFree(device_out);

  return out;
}
