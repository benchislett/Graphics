module Renderer

using ColorTypes
using LinearAlgebra

using ..GeometryTypes
using ..Cameras
using ..OBJ
using ..Intersections

export render

function render(scene::Hittable, camera::Camera, width::Integer, height::Integer)
  img = RGB.(zeros(width, height))

  for x in 1:width
    for y in 1:height
      u = x / width
      v = 1 - y / height

      ray = get_ray(camera, u, v)
      isect = intersection(scene, ray)

      if hit_test(isect)
        normal = hit_normal(isect)
        # normal = (1 .+ normal) ./ 2
        img[y, x] = RGB(normal...)
      end

    end
  end

  img
end

end