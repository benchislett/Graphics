#pragma once

#include "bxdf.cuh"
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
  Vector<int> material_ids;

  Vector<int> lights;
  Vector<DiffuseBRDF> diffuse_materials;

  __host__ void destroy() {
    triangles.destroy();
    normals.destroy();
    emissivities.destroy();
    material_ids.destroy();
    lights.destroy();
    diffuse_materials.destroy();
  }
};

struct DeviceScene;

struct HostScene : Scene<HostVector> {
  using Scene::Scene;

  HostScene& operator=(const DeviceScene& device_scene);
};

struct DeviceScene : Scene<DeviceVector> {
  DeviceScene& operator=(const HostScene& host_scene);
};

HostScene from_obj(const std::string& filename);
