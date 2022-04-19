module GeometryTypes

using StaticArrays, LinearAlgebra

export Vector3, Vector3f, Vector3i, Point3, Point3f, Point3i, Scalar
export Ray, Triangle, Sphere
export TriangleNormals, interpolate
export TriangleArray
export Hittable

const Scalar = Float32

const Vector3{T} = SVector{3,T} where {T<:Number}
const Vector3f = Vector3{Scalar}
const Vector3i = Vector3{Int32}

const Point3{T} = SVector{3,T} where {T<:Number}
const Point3f = Point3{Scalar}
const Point3i = Point3{Int32}

abstract type Hittable end

struct Ray
  origin::Point3f
  direction::Vector3f
end

struct Triangle <: Hittable
  vertices::SVector{3,Point3f}
end

Triangle(v1, v2, v3) = Triangle([v1, v2, v3])

struct TriangleNormals
  normals::SVector{3,Vector3f}
end

TriangleNormals(n1, n2, n3) = TriangleNormals([n1, n2, n3])
TriangleNormals(normal::Vector3f) = TriangleNormals([normal, normal, normal])
TriangleNormals(tri::Triangle) =
  TriangleNormals(normalize(cross(
    tri.vertices[2] - tri.vertices[1],
    tri.vertices[3] - tri.vertices[1]
  )))

interpolate(n::TriangleNormals, uvw::Vector3f) = sum(n.normals .* uvw)

struct Sphere <: Hittable
  center::Point3f
  radius::Scalar
end

struct TriangleArray <: Hittable
  triangles::Vector{Triangle}
  normals::Vector{TriangleNormals}
end

TriangleArray(triangles::Vector{Triangle}) =
  TriangleArray(triangles, map(TriangleNormals, triangles))

end