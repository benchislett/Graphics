module Intersections

using LinearAlgebra

using ..GeometryTypes

const hitepsilon = 0.001

export Intersection, TriangleIntersection, intersection, intersect_test

abstract type Intersection end

struct TriangleIntersection <: Intersection
  uvw::Vector3f
  time::Scalar
  hit::Bool
end

TriangleIntersection(hit::Bool) =
  TriangleIntersection(zero(Vector3f), 0.0, hit)

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

  uvw = [u, v, 1.0 - u - v]

  return TriangleIntersection(uvw, time, true)
end

function intersect_test(tri::Triangle, ray::Ray)::Bool
  return intersection(tri, ray).hit
end

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

  return SphereIntersection(time, true)
end

function intersect_test(sphere::Sphere, ray::Ray)::Bool
  return intersection(sphere, ray).hit
end

end