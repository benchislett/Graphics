#pragma once

#include "cu_misc.cuh"
#include "geometry.cuh"

#include <string>

template <template <typename> typename Vector>
struct Scene {
  int n_triangles;
  int n_lights;

  Vector<Triangle> triangles;
  Vector<TriangleNormal> normals;
  Vector<TriangleEmissivity> emissivities;

  Vector<int> lights;

  Scene(int n_t, int n_l)
      : n_triangles(n_t), n_lights(n_l), triangles{n_t}, normals{n_t}, emissivities{n_t}, lights{n_l} {}
};

struct DeviceScene;

struct HostScene : Scene<HostVector> {
  using Scene::Scene;

  HostScene& operator=(const DeviceScene& device_scene);
};

struct DeviceScene : Scene<DeviceVector> {
  using Scene::Scene;

  DeviceScene& operator=(const HostScene& host_scene);
};

HostScene from_obj(const std::string& filename);
