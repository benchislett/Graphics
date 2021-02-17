#include "scene.cuh"

HostScene& HostScene::operator=(const DeviceScene& device_scene) {
  triangles    = device_scene.triangles;
  normals      = device_scene.normals;
  emissivities = device_scene.emissivities;
  lights       = device_scene.lights;
  return *this;
}

DeviceScene& DeviceScene::operator=(const HostScene& host_scene) {
  triangles    = host_scene.triangles;
  normals      = host_scene.normals;
  emissivities = host_scene.emissivities;
  lights       = host_scene.lights;
  return *this;
}