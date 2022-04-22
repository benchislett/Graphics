module Intersections

using LinearAlgebra

using ..GeometryTypes
using ..SDFs
using ..OBJ

const hitepsilon = 0.001f0

export intersection
export Intersection, TriangleIntersection, TriangleArrayIntersection, SphereIntersection, SDFIntersection
export hit_test, hit_time, hit_point, hit_normal

abstract type Intersection end

struct TriangleIntersection <: Intersection
  uvw::Vector3f
  point::Point3f
  normal::Vector3f
  time::Scalar
  hit::Bool
end

TriangleIntersection() =
  TriangleIntersection(zero(Vector3f), zero(Vector3f), zero(Vector3f), 0.0, false)

function intersection(tri::Triangle, ray::Ray)::TriangleIntersection
  edge1 = tri.vertices[2] - tri.vertices[1]
  edge2 = tri.vertices[3] - tri.vertices[1]

  h = cross(ray.direction, edge2)
  determinant = dot(edge1, h)

  if abs(determinant) < eps(typeof(determinant))
    return TriangleIntersection()
  end

  invdeterminant = 1.0f0 / determinant
  s = ray.origin - tri.vertices[1]
  u = invdeterminant * dot(s, h)

  if u < 0.0f0 || u > 1.0f0
    return TriangleIntersection()
  end

  q = cross(s, edge1)
  v = invdeterminant * dot(ray.direction, q)

  if v < 0.0f0 || u + v > 1.0f0
    return TriangleIntersection()
  end

  time = invdeterminant * dot(edge2, q)

  if time < hitepsilon
    return TriangleIntersection()
  end

  point = ray.origin + time * ray.direction
  normal = normalize(cross(edge1, edge2))
  uvw = Vector3f(u, v, 1.0f0 - u - v)

  TriangleIntersection(uvw, point, normal, time, true)
end

struct TriangleArrayIntersection <: Intersection
  uvw::Vector3f
  point::Point3f
  normal::Vector3f
  time::Scalar
  hit::Bool
  which::Int32
end

TriangleArrayIntersection() =
  TriangleArrayIntersection(zero(Vector3f), zero(Point3f), zero(Vector3f), 0.0, false, 0)

function intersection(array::TriangleArray, ray::Ray)::TriangleArrayIntersection
  closest = TriangleArrayIntersection()
  for i in eachindex(array.triangles)
    tri = array.triangles[i]
    isect = intersection(tri, ray)
    if !hit_test(isect)
      continue
    elseif !hit_test(closest) || hit_time(isect) < hit_time(closest)
      normal = interpolate(array.normals[i], isect.uvw)
      closest = TriangleArrayIntersection(isect.uvw, isect.point, normal, isect.time, isect.hit, i)
    end
  end

  closest
end

struct SphereIntersection <: Intersection
  point::Point3f
  normal::Vector3f
  time::Scalar
  hit::Bool
end

SphereIntersection() = SphereIntersection(zero(Point3f), zero(Vector3f), 0.0, false)

function intersection(sphere::Sphere, ray::Ray)::SphereIntersection
  oc = ray.origin - sphere.center
  a = dot(ray.direction, ray.direction)
  b = 2.0 * dot(oc, ray.direction)
  c = dot(oc, oc) - sphere.radius^2

  discriminant = b^2 - 4a * c

  if discriminant < 0
    return SphereIntersection()
  end

  root1 = (-b - sqrt(discriminant)) / 2a
  root2 = (-b + sqrt(discriminant)) / 2a

  if root2 < 0
    return SphereIntersection()
  end

  time = root2
  if root1 >= 0
    time = root1
  end

  if time < hitepsilon
    return SphereIntersection()
  end

  point = ray.origin + time * ray.direction
  normal = normalize(point - sphere.center)
  SphereIntersection(point, normal, time, true)
end

struct SDFIntersection <: Intersection
  point::Point3f
  normal::Vector3f
  time::Scalar
  hit::Bool
end

SDFIntersection() = SDFIntersection(zero(Point3f), zero(Vector3f), 0.0, false)

# Sphere Tracing
function intersection(f::SDF, ray::Ray)
  e = 0.00001
  dist = Inf32
  time = 10 * hitepsilon
  point = ray.origin + time * ray.direction
  steps = 0
  while abs(dist) > e
    dist = sample(f, point)
    time += dist
    point += dist * ray.direction
    steps += 1

    if steps > 10000
      return SDFIntersection()
    end
  end

  e = 0.001
  dx = Point3f(e, 0, 0)
  dy = Point3f(0, e, 0)
  dz = Point3f(0, 0, e)
  xdev = sample(f, point + dx) - sample(f, point - dx)
  ydev = sample(f, point + dy) - sample(f, point - dy)
  zdev = sample(f, point + dz) - sample(f, point - dz)
  normal = normalize(Vector3f(xdev, ydev, zdev))

  SDFIntersection(point, normal, time, true)
end

hit_test(isect::Intersection) = isect.hit
hit_time(isect::Intersection) = isect.time
hit_point(isect::Intersection) = isect.point
hit_normal(isect::Intersection) = isect.normal

end