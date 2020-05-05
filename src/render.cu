#include "render.cuh"
#include "intersection.cuh"
#include "path.cuh"
#include "random.cuh"

__global__ void render_kernel(RenderParams params, Scene scene, Image im) {
  int xid = blockDim.x * blockIdx.x + threadIdx.x;
  int yid = blockDim.y * blockIdx.y + threadIdx.y;

  if (xid >= im.width || yid >= im.height) {
    return;
  }

  int tid = yid * im.width + xid;

  LocalDeviceRNG local_gen = scene.gen.local(tid);

  Vec3 colour(0.f);
  Vec3 colour_tmp;
  float u, v;
  for (int i = 0; i < params.spp; i++) {
    u = ((float)xid + local_gen.generate()) / (float)(im.width);
    v = ((float)yid + local_gen.generate()) / (float)(im.height);
    
    colour_tmp = trace(scene.cam.get_ray(u, v), scene, local_gen, 50);
    colour_tmp.e[0] = fmax(fmin(colour_tmp.e[0], 1.f), 0.f);
    colour_tmp.e[1] = fmax(fmin(colour_tmp.e[1], 1.f), 0.f);
    colour_tmp.e[2] = fmax(fmin(colour_tmp.e[2], 1.f), 0.f);
    colour += colour_tmp;
  }
  colour /= (float)(params.spp);

  im.film[tid] = colour;
}

__host__ void render(const RenderParams &params, Scene &scene, Image &im) {
  scene.to_device();
  im.to_device();

  int nx = im.width;
  int ny = im.height;

  if (scene.gen.state == NULL) scene.gen = DeviceRNG(nx * ny);
  else printf("Scene DeviceRNG already initialized!\n");

  dim3 threads(16, 16);
  dim3 blocks(nx / threads.x + 1, ny / threads.y + 1);

  printf("rendering...\n");
  render_kernel<<<blocks, threads>>>(params, scene, im);
  // render_kernel<<<1, 1>>>(params, scene, im);
  cudaCheckError();
  cudaDeviceSynchronize();
  printf("done\n");

  scene.to_host();
  im.to_host();
}
