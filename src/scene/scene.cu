#include "scene.cuh"

HostScene& HostScene::operator=(const DeviceScene& device_scene) {
  n_triangles = device_scene.n_triangles;
  n_lights    = device_scene.n_lights;

  triangles         = device_scene.triangles;
  normals           = device_scene.normals;
  emissivities      = device_scene.emissivities;
  lights            = device_scene.lights;
  material_ids      = device_scene.material_ids;
  diffuse_materials = device_scene.diffuse_materials;
  return *this;
}

DeviceScene& DeviceScene::operator=(const HostScene& host_scene) {
  n_triangles = host_scene.n_triangles;
  n_lights    = host_scene.n_lights;

  triangles         = host_scene.triangles;
  normals           = host_scene.normals;
  emissivities      = host_scene.emissivities;
  lights            = host_scene.lights;
  material_ids      = host_scene.material_ids;
  diffuse_materials = host_scene.diffuse_materials;
  return *this;
}
