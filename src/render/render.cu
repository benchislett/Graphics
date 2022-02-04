#include "bvh.cuh"
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

struct Path {
  Ray cur;
  float3 L;
  int px;
  int py;
  bool active;
};

__global__ void init_paths(Vector<Path> pq, Camera cam, unsigned int w, unsigned int h, unsigned int spp,
                           unsigned int paths_processed) {
  unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  int pp = paths_processed + idx;

  if (idx >= pq.size)
    return;

  int spp_rt = (int) sqrtf((float) spp);

  int sample = pp / (w * h);
  int coord  = pp % (w * h);
  int x      = coord % w;
  int y      = coord / w;
  int sx     = sample % spp_rt;
  int sy     = sample / spp_rt;

  float u = ((float) x + (float) sx / (float) spp_rt) / (float) w;
  float v = ((float) y + (float) sy / (float) spp_rt) / (float) h;

  Ray r = cam.get_ray(u, v);

  pq[idx] = {r, {0, 0, 0}, x, y, true};
}

__global__ void advance_paths(BVH bvh, Vector<Path> pq, Image out, float spp) {
  unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (idx >= pq.size)
    return;

  Path p = pq[idx];
  Ray r  = p.cur;

  auto i = bvh.intersects(r);

  if (i.hit) {
    pq[idx].cur = Ray(i.point, i.normal);
    Vec3 normal = i.normal;
    // // p.L += normal;
    p.L.x += (normal.x + 1.0) / 2.0;
    p.L.y += (normal.y + 1.0) / 2.0;
    p.L.z += (normal.z + 1.0) / 2.0;
  }

  out[p.py * out.width + p.px] += p.L / spp;
}

Image render_normals(TriangleArray tris, Camera cam, unsigned int w, unsigned int h) {

  Image out(w, h);

  for (int i = 0; i < w * h; i++) {
    out[i] = {0, 0, 0};
  }

  BVH bvh(tris);

  ScopedMicroTimer x_([&](int us) { printf("Rendered in %.2f ms\n", (double) us / 1000.0); });

  unsigned int spp = 1;

  unsigned int total_paths = w * h * spp;

  unsigned int path_queue_size = min(total_paths, 1024 * 1024 * 32);
  Vector<Path> path_queue(path_queue_size);

  unsigned int paths_processed = 0, rounds = 0;
  while (paths_processed < total_paths) {
    dim3 block(128);
    dim3 grid((path_queue_size + 127) / 128);
    init_paths<<<grid, block>>>(path_queue, cam, w, h, spp, paths_processed);
    cudaDeviceSynchronize();
    cudaCheckError();
    // for (int idx = 0; idx < path_queue_size; idx++) {
    //   Path p = path_queue[idx];
    //   Ray r  = p.cur;

    //   auto i = bvh.intersects(r);

    //   if (i.hit) {
    //     Vec3 normal = i.normal;
    //     // // p.L += normal;
    //     p.L.x += (normal.x + 1.0) / 2.0;
    //     p.L.y += (normal.y + 1.0) / 2.0;
    //     p.L.z += (normal.z + 1.0) / 2.0;
    //   }

    //   out[p.py * out.width + p.px] += p.L / spp;
    // }
    advance_paths<<<grid, block>>>(bvh, path_queue, out, (float) spp);
    cudaDeviceSynchronize();
    cudaCheckError();

    rounds++;
    paths_processed += path_queue_size;
  }

  return out;
}
