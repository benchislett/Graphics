module Intersections

using LinearAlgebra

using ..GeometryTypes

const hitepsilon = 0.001

export Intersection, TriangleIntersection, intersection, intersect_test

abstract type Intersection end

struct TriangleIntersection <: Intersection
  point::Point3f
  uvw::Vector3f
  time::Float32
  hit::Bool
end

TriangleIntersection(hit::Bool) =
  TriangleIntersection(zero(Point3f), zero(Vector3f), 0.0, hit)

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
  uvw = [u, v, 1.0 - u - v]

  return TriangleIntersection(point, uvw, time, true)
end

function intersect_test(tri::Triangle, ray::Ray)::Bool
  return intersection(tri, ray).hit
end

end