module Intersections

using LinearAlgebra

using ..GeometryTypes
using ..OBJ

const hitepsilon = 0.001

export Intersection, TriangleIntersection
export intersection
export hit_test, hit_time, hit_point, hit_normal

abstract type Intersection end

struct TriangleIntersection <: Intersection
  uvw::Vector3f
  point::Point3f
  normal::Vector3f
  time::Scalar
  hit::Bool
end

TriangleIntersection(hit::Bool) =
  TriangleIntersection(zero(Vector3f), zero(Vector3f), zero(Vector3f), 0.0, hit)

function intersection(tri::Triangle, ray::Ray)::TriangleIntersection
  edge1 = tri.vertices[2] - tri.vertices[1]
  edge2 = tri.vertices[3] - tri.vertices[1]

  h = cross(ray.direction, edge2)
  determinant = dot(edge1, h)

  if abs(determinant) < eps(typeof(determinant))
    return TriangleIntersection(false)
  end

  invdeterminant = 1.0 / determinant
  s = ray.origin - tri.vertices[1]
  u = invdeterminant * dot(s, h)

  if u < 0.0 || u > 1.0
    return TriangleIntersection(false)
  end

  q = cross(s, edge1)
  v = invdeterminant * dot(ray.direction, q)

  if v < 0.0 || u + v > 1.0
    return TriangleIntersection(false)
  end

  time = invdeterminant * dot(edge2, q)

  if time < hitepsilon
    return TriangleIntersection(false)
  end

  point = ray.origin + time * ray.direction
  normal = TriangleNormals(tri).normals[1]
  uvw = [u, v, 1.0 - u - v]

  TriangleIntersection(uvw, point, normal, time, true)
end

function intersection(scene::OBJMeshScene, ray::Ray)::TriangleIntersection
  closest = TriangleIntersection(false)
  for i in eachindex(scene.triangles)
    tri = scene.triangles[i]
    isect = intersection(tri, ray)
    if !hit_test(isect)
      continue
    elseif !hit_test(closest) || hit_time(isect) < hit_time(closest)
      normal = interpolate(scene.normals[i], isect.uvw)
      closest = TriangleIntersection(isect.uvw, isect.point, normal, isect.time, isect.hit)
    end
  end

  closest
end

hit_test(isect::TriangleIntersection) = isect.hit
hit_time(isect::TriangleIntersection) = isect.time
hit_point(isect::TriangleIntersection) = isect.point
hit_normal(isect::TriangleIntersection) = isect.normal

struct SphereIntersection <: Intersection
  time::Scalar
  hit::Bool
end

SphereIntersection(hit::Bool) = SphereIntersection(0.0, hit)

function intersection(sphere::Sphere, ray::Ray)::SphereIntersection
  oc = ray.origin - sphere.center
  a = dot(ray.direction, ray.direction)
  b = 2.0 * dot(oc, ray.direction)
  c = dot(oc, oc) - sphere.radius^2

  discriminant = b^2 - 4a * c

  if discriminant < 0
    return SphereIntersection(false)
  end

  root1 = (-b - sqrt(discriminant)) / 2a
  root2 = (-b + sqrt(discriminant)) / 2a

  if root2 < 0
    return SphereIntersection(false)
  end

  time = root2
  if root1 >= 0
    time = root1
  end

  SphereIntersection(time, true)
end

hit_test(isect::SphereIntersection) = isect.hit
hit_time(isect::SphereIntersection) = isect.time

end