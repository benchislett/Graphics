#include "cu_rand.cuh"
#include "integrate.cuh"

#include <iostream>

__device__ float3 trace(const Ray ray, DeviceScene& scene) {
  int which;

  TriangleHitRecord record = first_hit(ray, scene.triangles.data, scene.n_triangles, &which);
  if (record.hit) {
    int which_vis;
    float3 hit_point  = interp(scene.triangles[which], record.u, record.v);
    float3 hit_normal = interp(scene.normals[which], record.u, record.v);
    float3 target =
        scene.triangles[scene.lights[0]].v0 + scene.triangles[scene.lights[0]].v1 + scene.triangles[scene.lights[0]].v2;
    target /= 3.f;
    Ray vis_ray                  = (Ray){hit_point, normalized(target - hit_point)};
    TriangleHitRecord vis_record = first_hit(vis_ray, scene.triangles.data, scene.n_triangles, &which_vis);
    if (vis_record.hit && which_vis == scene.lights[0]) {
      return scene.diffuse_materials[0].albedo * scene.emissivities[scene.lights[0]].intensity / vis_record.time
           * cosf(dot(vis_ray.direction, hit_normal));
    }
  }
  return make_float3(0.f);
}

__global__ void render_kernel(const Camera camera, DeviceScene scene, DeviceRNG rng, uint3 param, uchar4* pixels) {
  unsigned int i   = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int j   = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int tid = i * param.y + j;

  if (i >= param.x || j >= param.y)
    return;

  unsigned int spp = param.z;

  float3 val = make_float3(0.f);
  for (int k = 0; k < spp; k++) {
    float u = ((float) j + rng.uniform(tid)) / (float) param.y;
    float v = ((float) i + rng.uniform(tid)) / (float) param.x;

    Ray ray = get_ray(camera, u, v);
    val += trace(ray, scene);
  }
  val /= (float) spp;
  val = clamp(val, 0.f, 1.f);
  val *= 255.9999f;

  uchar4 pixel = make_uchar4((unsigned char) val.x, (unsigned char) val.y, (unsigned char) val.z, (unsigned char) 255);

  pixels[tid] = pixel;
}

Image render(const Camera camera, DeviceScene& scene, int x, int y, int spp) {
  // cache on same image size
  static DeviceRNG rng;
  static int last_x = 0;
  static int last_y = 0;
  if (x != last_x || y != last_y) {
    rng    = init_rng(x * y);
    last_x = x;
    last_y = y;
  }

  Image image;
  image.x    = x;
  image.y    = y;
  image.data = (uchar4*) malloc(x * y * sizeof(uchar4));

  cudaCheckError();
  uchar4* gpu_data;
  cudaMalloc((void**) &gpu_data, x * y * sizeof(uchar4));
  cudaCheckError();

  dim3 threads(16, 16);
  dim3 blocks(x / threads.x + 1, y / threads.y + 1);
  render_kernel<<<blocks, threads>>>(camera, scene, rng, make_uint3(x, y, spp), gpu_data);
  cudaDeviceSynchronize();
  cudaCheckError();

  cudaMemcpy(image.data, gpu_data, x * y * sizeof(uchar4), cudaMemcpyDeviceToHost);
  cudaCheckError();

  cudaFree(gpu_data);
  cudaCheckError();

  return image;
}
