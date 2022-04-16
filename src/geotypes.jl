module GeometryTypes

using StaticArrays

export Vector3, Vector3f, Vector3i, Point3, Point3f, Point3i, Scalar
export Ray, Triangle, Sphere
export Scene

const Scalar = Float32

const Vector3{T} = SVector{3,T} where {T<:Number}
const Vector3f = Vector3{Scalar}
const Vector3i = Vector3{Int64}

const Point3{T} = SVector{3,T} where {T<:Number}
const Point3f = Point3{Scalar}
const Point3i = Point3{Int64}

struct Ray
  origin::Point3f
  direction::Vector3f
end

struct Triangle
  vertices::SVector{3,Point3f}
end

Triangle(v1, v2, v3) = Triangle([v1, v2, v3])

struct Sphere
  center::Point3f
  radius::Scalar
end

abstract type Scene end

end