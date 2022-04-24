module Renderer

using ColorTypes
using LinearAlgebra
using Statistics
using CUDA

using ..GeometryTypes
using ..Cameras
using ..OBJ
using ..Intersections

export render!, render

blankimg(width, height) = RGB.(zeros(Scalar, width, height))

function sample_sphere()::Vector3f
  u, v = rand(Scalar), rand(Scalar)

  z = 1 - 2u
  r = sqrt(max(0, 1 - z^2))
  Φ = 2π * v
  normalize(Vector3f(r * cos(Φ), r * sin(Φ), z))
end

function sample_pointed_hemisphere(n)::Vector3f
  w = zero(Vector3f)
  while dot(w, n) <= 0.001
    w = sample_sphere()
  end

  w
end

function trace_and_shade(scene::Hittable, camera::Camera, u::Float32, v::Float32)
  ray = get_ray(camera, u, v)
  isect = intersection(scene, ray)

  illum::Vector3f = zero(Vector3f)

  if hit_test(isect)
    normal = hit_normal(isect)

    # Simple normal-based illumination
    # illum = (1 .+ normal) ./ 2

    # Ambient Occlusion
    nsamples = 8
    occlusion = 0.0f0
    point = hit_point(isect)
    for i in 1:nsamples
      w = sample_pointed_hemisphere(normal)
      isect = intersection(scene, Ray(point, w))
      if !hit_test(isect)
        occlusion += dot(w, normal)
      end
    end

    pdf = 1.0f0 / (2π)
    occlusion /= π
    occlusion /= pdf * nsamples
    illum = Vector3f(occlusion, occlusion, occlusion)

    if any(isnan.(illum))
      illum = Vector3f(1, 1, 1)
    end
  end

  RGB(illum...)
end

function rendercpu!(scene::Hittable, camera::Camera, img)
  width, height = size(img)
  for x in 1:width
    for y in 1:height
      u::Float32 = x / width
      v::Float32 = 1 - y / height

      img[y, x] = trace_and_shade(scene, camera, u, v)
    end
  end

  img
end

function rendergpukernel!(scene, camera, gpuimg)
  width::Int32, height::Int32 = size(gpuimg)
  x::Int32 = threadIdx().x
  y::Int32 = blockIdx().x
  u::Float32 = x / width
  v::Float32 = 1 - y / height

  # @cuprintf("%d %d %d %d\n", x, y, u, v)
  gpuimg[y, x] = trace_and_shade(scene, camera, u, v)

  nothing
end

function rendergpu!(scene::Hittable, camera::Camera, img)
  x, y = size(img)
  gpuimg = cu(img)
  @cuda threads = x blocks = y rendergpukernel!(scene, camera, gpuimg)
  copyto!(img, gpuimg)
end

render(scene, camera, width, height) = rendergpu!(scene, camera, blankimg(width, height))

end