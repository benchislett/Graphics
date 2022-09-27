module GeometryPrimitives

import StaticArrays.SVector, StaticArrays.@SVector

using ..GeometryCore

export Geometric3DPrimitive
export Triangle, Sphere, AABB
export Ray

abstract type Geometric3DPrimitive end

struct Triangle <: Geometric3DPrimitive
  vertices::SVector{3,Point3f}
end

Triangle(v1, v2, v3) = Triangle(@SVector [v1, v2, v3])

struct Sphere <: Geometric3DPrimitive
  center::Point3f
  radius::Scalar
end

# Axis-Aligned Bounding Box
# struct AABB <: Geometric3DPrimitive
#   min::Vector3f
#   max::Vector3f
# end

struct Ray
  origin::Point3f
  direction::UnitVector3f
end

end