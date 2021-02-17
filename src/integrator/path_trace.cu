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
      return scene.emissivities[scene.lights[0]].intensity / vis_record.time * cosf(dot(vis_ray.direction, hit_normal));
    }
  }
  return make_float3(0.f);
}

__global__ void render_kernel(const Camera camera, DeviceScene scene, uint3 param, uchar4* pixels) {
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= param.x || j >= param.y)
    return;

  float u = (float) j / (float) param.y;
  float v = (float) i / (float) param.x;
  // unsigned int spp = param.z;


  Ray ray    = get_ray(camera, u, v);
  float3 val = trace(ray, scene);
  val        = clamp(val, 0.f, 1.f);
  val *= 255.9999f;

  uchar4 pixel = make_uchar4((unsigned char) val.x, (unsigned char) val.y, (unsigned char) val.z, (unsigned char) 255);

  pixels[i * param.y + j] = pixel;
}

Image render(const Camera camera, DeviceScene& scene, int x, int y, int spp) {
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
  render_kernel<<<blocks, threads>>>(camera, scene, make_uint3(x, y, spp), gpu_data);
  cudaDeviceSynchronize();
  cudaCheckError();

  cudaMemcpy(image.data, gpu_data, x * y * sizeof(uchar4), cudaMemcpyDeviceToHost);
  cudaCheckError();

  cudaFree(gpu_data);
  cudaCheckError();

  return image;
}
