module Renderer

using ColorTypes
using LinearAlgebra
using Statistics
using CUDA

using ..GeometryTypes
using ..Cameras
using ..OBJ
using ..Intersections
using ..Sampling

export Scene, Integrator, NormalsIntegrator, AOIntegrator, render!

struct Scene{T<:Hittable}
  geometry::T
  camera::Camera
  film::Matrix{RGB{Scalar}}
end

abstract type Integrator end

struct NormalsIntegrator <: Integrator end
struct AOIntegrator <: Integrator
  nsamples::Int32
end

blankimg(width, height) = RGB.(zeros(Scalar, width, height))
Scene(geometry::Hittable, camera::Camera, width, height) = Scene(geometry, camera, blankimg(width, height))

function trace_and_shade(scene::Scene, ::NormalsIntegrator, u::Scalar, v::Scalar)::RGB{Scalar}
  ray = get_ray(scene.camera, u, v)
  isect = intersection(scene.geometry, ray)

  illum::Vector3f = zero(Vector3f)

  if hit_test(isect)
    normal = hit_normal(isect)

    # Simple normal-based illumination
    illum = (1 .+ normal) ./ 2

    if any(isnan.(illum))
      illum = Vector3f(1, 1, 1)
    end
  end

  RGB(illum...)
end

function trace_and_shade(scene::Scene, integrator::AOIntegrator, u::Scalar, v::Scalar)::RGB{Scalar}
  ray = get_ray(scene.camera, u, v)
  isect = intersection(scene.geometry, ray)

  illum::Vector3f = zero(Vector3f)

  if hit_test(isect)
    normal = hit_normal(isect)

    # Ambient Occlusion
    nsamples = integrator.nsamples
    occlusion = 0.0f0
    point = hit_point(isect)
    for i in 1:nsamples
      w = sample_oriented_hemisphere(normal, rand(Scalar), rand(Scalar))
      isect = intersection(scene.geometry, Ray(point, w))
      if !hit_test(isect)
        occlusion += dot(w, normal)
      end
    end

    pdf = 1.0f0 / 2π
    occlusion /= π
    occlusion /= pdf * nsamples
    illum = Vector3f(occlusion, occlusion, occlusion)

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