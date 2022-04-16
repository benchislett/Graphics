module Renderer

using ColorTypes

using ..GeometryTypes
using ..Cameras
using ..OBJ
using ..Intersections

export render

function render(scene::OBJMeshScene, camera::Camera, width::Int, height::Int)
  img = RGB.(zeros(width, height))

  for x in 1:width
    for y in 1:height
      u = x / width
      v = y / height

      ray = get_ray(camera, u, v)
      isect = intersection(scene, ray)

      if hit_test(isect)
        img[x, y] = RGB(1, 0, 0)
      end

    end
  end

  img
end

end