module Renderer

using ColorTypes
using LinearAlgebra
using Statistics

using ..GeometryCore
using ..GeometryMeshes
using ..Cameras
using ..OBJ
using ..GeometryIntersection
using ..Sampling
using ..Materials

export Scene, Integrator
export NormalsIntegrator, DepthIntegrator
export render!

struct Scene
  geometry::TriangleMesh
  materials::Vector{Material}
  light::Int32
  camera::Camera
  film::Matrix{RGB{Scalar}}
end

blankimg(width, height) = RGB.(zeros(Scalar, width, height))
Scene(geometry, materials, light, camera, width, height) = Scene(geometry, materials, light, camera, blankimg(width, height))

abstract type Integrator end

struct NormalsIntegrator <: Integrator end
struct DepthIntegrator <: Integrator end

function shade_intersection(scene, ::NormalsIntegrator, isect)
  normal = isect.normal
  (1 .+ normal) ./ 2
end

function shade_intersection(scene, ::DepthIntegrator, isect)
  depth = isect.time
  i = 1 / depth
  Vector3f(i, i, i)
end

function trace_and_shade(scene, integrator, u, v)
  ray = get_ray(scene.camera, u, v)
  isect = intersection(scene.geometry, ray)

  illum = zero(Vector3f)

  if !ismissing(isect)
    illum = shade_intersection(scene, integrator, isect)

    if any(isnan.(illum))
      illum = Vector3f(1, 1, 1)
    end
  end

  RGB(illum...)
end

function rendercpu!(scene::Scene, integrator::Integrator)
  width, height = size(scene.film)
  for x in 1:width
    for y in 1:height
      u::Scalar = x / width
      v::Scalar = 1 - y / height
      scene.film[y, x] = trace_and_shade(scene, integrator, u, v)
    end
  end

  scene.film
end

render!(scene, integrator) = rendercpu!(scene, integrator)

end