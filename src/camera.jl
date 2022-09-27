module Cameras

using LinearAlgebra

using ..GeometryCore
using ..GeometryPrimitives

begin
  const viewup = Vector3f(0, 1, 0)
end

export Camera, PerspectiveCamera, OrthographicCamera, get_ray

abstract type Camera end

struct PerspectiveCamera <: Camera
  position::Point3f
  lower_left::Point3f
  horizontal::Vector3f
  vertical::Vector3f

  function PerspectiveCamera(aspect, fov, position, target)
    bh = 2.0 * tan(fov / 2.0)
    bw = bh * aspect

    w = normalize(position - target)
    u = normalize(cross(viewup, w))
    v = cross(w, u)

    lower_left = position - (bw * u / 2.0) - (bh * v / 2.0) - w
    horizontal = u * bw
    vertical = v * bh

    new(position, lower_left, horizontal, vertical)
  end
end

function get_ray(camera::PerspectiveCamera, u, v)::Ray
  target = camera.lower_left + (u * camera.horizontal) + (v * camera.vertical)
  direction = normalize(target - camera.position)

  Ray(camera.position, direction)
end

struct OrthographicCamera <: Camera
  center::Point3f
  horizontal::Vector3f
  vertical::Vector3f
  direction::Vector3f

  OrthographicCamera(center, horizontal, vertical, target) =
    new(center, horizontal, vertical, normalize(target - center))
end

function get_ray(camera::OrthographicCamera, u, v)::Ray
  position = camera.center + (camera.horizontal * (u - 0.5)) + (camera.vertical * (v - 0.5))
  Ray(position, camera.direction)
end

end