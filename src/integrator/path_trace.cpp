#include "integrate.h"

#include <iostream>

float3 trace(const Ray ray, const Scene scene) {
  int which;
  TriangleHitRecord record = first_hit(ray, scene.triangles, scene.n_triangles, &which);
  if (record.hit) {
    int which_vis;
    float3 hit_point  = interp(scene.triangles[which], record.u, record.v);
    float3 hit_normal = interp(scene.normals[which], record.u, record.v);
    float3 target =
        scene.triangles[scene.lights[0]].v0 + scene.triangles[scene.lights[0]].v1 + scene.triangles[scene.lights[0]].v2;
    target /= 3.f;
    Ray vis_ray                  = (Ray){hit_point, normalized(target - hit_point)};
    TriangleHitRecord vis_record = first_hit(vis_ray, scene.triangles, scene.n_triangles, &which_vis);
    if (vis_record.hit && which_vis == scene.lights[0]) {
      return scene.emissivities[scene.lights[0]].intensity / vis_record.time * cosf(dot(vis_ray.direction, hit_normal));
    }
  }
  return make_float3(0.f);
}

Image render(const Camera camera, const Scene scene, int x, int y, int spp) {
  Image image;
  image.x    = x;
  image.y    = y;
  image.data = (uchar4*) malloc(x * y * sizeof(uchar4));

  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      float3 val = make_float3(0.f);
      float u    = (float) j / (float) y;
      float v    = (float) i / (float) x;
      for (int s = 0; s < spp; s++) {
        Ray ray = get_ray(camera, u, v);
        val += trace(ray, scene);
      }
      val /= (float) spp;
      val = clamp(val, 0.f, 1.f);
      val *= 255.9999f;
      uchar4 data;
      data.x = (unsigned char) val.x;
      data.y = (unsigned char) val.y;
      data.z = (unsigned char) val.z;
      data.w = (unsigned char) 255;

      image.data[i * y + j] = data;
    }
  }

  return image;
}
