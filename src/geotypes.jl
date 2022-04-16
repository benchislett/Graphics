module GeometryTypes

using StaticArrays

export Vector3, Vector3f, Vector3i, Point3, Point3f, Point3i, Ray

const Vector3{T} = SVector{3,T} where {T<:Number}
const Vector3f = Vector3{Float32}
const Vector3i = Vector3{Int32}

const Point3{T} = SVector{3,T} where {T<:Number}
const Point3f = Point3{Float32}
const Point3i = Point3{Int32}

struct Ray
  origin::Point3f
  direction::Vector3f
end

end