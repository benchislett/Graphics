module Renderer

using ColorTypes
using LinearAlgebra
using Statistics

using ..GeometryTypes
using ..Cameras
using ..OBJ
using ..Intersections

export render!, render

blankimg(width, height) = RGB.(zeros(Scalar, width, height))

function sample_sphere()
  u, v = rand(Scalar), rand(Scalar)

  z = 1 - 2u
  r = sqrt(max(0, 1 - z^2))
  Φ = 2π * v
  normalize(Vector3f(r * cos(Φ), r * sin(Φ), z))
end

function sample_pointed_hemisphere(n)
  w = zero(Vector3f)
  while dot(w, n) <= 0.001
    w = sample_sphere()
  end

  w
end

function render!(scene::Hittable, camera::Camera, img)
  width, height = size(img)
  for x in 1:width
    for y in 1:height
      u = x / width
      v = 1 - y / height

      ray = get_ray(camera, u, v)
      isect = intersection(scene, ray)

      if hit_test(isect)
        normal = hit_normal(isect)

        # Simple normal-based illumination
        illum = (1 .+ normal) ./ 2

        # Ambient Occlusion
        # nsamples = 32
        # occlusion = 0.0f0
        # point = hit_point(isect)
        # for i in 1:nsamples
        #   w = sample_pointed_hemisphere(normal)
        #   isect = intersection(scene, Ray(point, w))
        #   if !hit_test(isect)
        #     occlusion += dot(w, normal)
        #   end
        # end

        # pdf = 1.0f0 / (2π)
        # occlusion /= π
        # occlusion /= pdf * nsamples
        # illum = ntuple(_ -> occlusion, 3)

        if any(isnan.(illum))
          illum = [1, 1, 1]
        end
        img[y, x] = RGB(illum...)
      end

    end
  end

  img
end

render(scene, camera, width, height) = render!(scene, camera, blankimg(width, height))

end