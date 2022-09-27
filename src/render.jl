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

export Scene, Integrator, NormalsIntegrator, AOIntegrator, ShadowIntegrator, render!

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
struct AOIntegrator <: Integrator
  nsamples::Int32
end

struct ShadowIntegrator <: Integrator end

function trace_and_shade(scene::Scene, ::NormalsIntegrator, u::Scalar, v::Scalar)::RGB{Scalar}
  ray = get_ray(scene.camera, u, v)
  isect = intersection(scene.geometry, ray)

  illum::Vector3f = zero(Vector3f)

  if !ismissing(isect)
    normal = isect.normal

    # Simple normal-based illumination
    illum = (1 .+ normal) ./ 2

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
      u::Float32 = x / width
      v::Float32 = 1 - y / height
      scene.film[y, x] = trace_and_shade(scene, integrator, u, v)
    end
  end

  scene.film
end

render!(scene, integrator) = rendercpu!(scene, integrator)

end