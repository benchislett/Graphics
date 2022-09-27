module GeometryIntersection

using LinearAlgebra

using ..GeometryCore
using ..GeometryPrimitives
using ..GeometryMeshes

export PrimitiveIntersection, min
export intersection

struct PrimitiveIntersection
  point::Point3f
  time::Scalar
  normal::UnitVector3f
  uv::Vector2f
end

min(isect1::PrimitiveIntersection, isect2::PrimitiveIntersection) =
  (isect1.time <= isect2.time) ? isect1 : isect2

function intersection(tri::Triangle, ray::Ray)
  edge1 = tri.vertices[2] - tri.vertices[1]
  edge2 = tri.vertices[3] - tri.vertices[1]

  h = cross(ray.direction, edge2)
  determinant = dot(edge1, h)

  if abs(determinant) < eps(typeof(determinant))
    return missing
  end

  invdeterminant = 1.0 / determinant
  s = ray.origin - tri.vertices[1]
  u = invdeterminant * dot(s, h)

  if u < 0.0 || u > 1.0
    return missing
  end

  q = cross(s, edge1)
  v = invdeterminant * dot(ray.direction, q)

  if v < 0.0f0 || u + v > 1.0
    return missing
  end

  time = invdeterminant * dot(edge2, q)

  if time < HitEpsilon
    return missing
  end

  point = ray.origin + time * ray.direction
  normal = normalize(cross(edge1, edge2))
  uv = Vector2f(u, v)

  PrimitiveIntersection(point, time, normal, uv)
end

function intersection(sphere::Sphere, ray::Ray)
  oc = ray.origin - sphere.center
  a = dot(ray.direction, ray.direction)
  b = 2.0 * dot(oc, ray.direction)
  c = dot(oc, oc) - sphere.radius^2

  discriminant = b^2 - 4a * c

  if discriminant < 0
    return missing
  end

  root1 = (-b - sqrt(discriminant)) / 2a
  root2 = (-b + sqrt(discriminant)) / 2a

  if root2 < 0
    return missing
  end

  time = root2
  if root1 >= 0
    time = root1
  end

  if time < HitEpsilon
    return missing
  end

  point = ray.origin + time * ray.direction
  normal = normalize(point - sphere.center)

  # Cylindrical projection
  u = atan(normal.x, normal.z) / 2π + 0.5
  v = normal.y * 0.5 + 0.5
  uv = Vector2f(u, v)

  PrimitiveIntersection(point, time, normal, uv)
end

function intersection(mesh::TriangleMesh, ray::Ray)
  isect = missing
  which = 0
  for i in eachindex(mesh)
    tmpisect = intersection(mesh[i], ray)
    if ismissing(isect) || (!ismissing(tmpisect) && tmpisect.time < isect.time)
      isect = tmpisect
      which = i
    end
  end

  if !ismissing(isect) && length(mesh.normals) > 0
    normals = mesh.normals[mesh.faceindices[which]]
    u, v = isect.uv
    uvw = [u, v, 1 - u - v]
    normal = sum(normals .* uvw)
    isect = PrimitiveIntersection(isect.point, isect.time, normal, isect.uv)
  end

  isect
end

end