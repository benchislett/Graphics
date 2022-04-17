module Renderer

using ColorTypes

using ..GeometryTypes
using ..Cameras
using ..OBJ
using ..Intersections

export render

function render(scene::Scene, camera::Camera, width::Integer, height::Integer)
  img = RGB.(zeros(width, height))

  for x in 1:width
    for y in 1:height
      u = x / width
      v = 1 - y / height

      ray = get_ray(camera, u, v)
      isect = intersection(scene, ray)

      if hit_test(isect)
        normal = hit_normal(isect)
        img[y, x] = RGB(normal...)
      end

    end
  end

  img
end

end