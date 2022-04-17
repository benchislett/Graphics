module SDFs

using ..GeometryTypes
using LinearAlgebra

export SDF, SphereSDF, CubeSDF
export sample

abstract type SDF <: Hittable end

struct SphereSDF <: SDF
  origin::Point3f
  radius::Scalar
end

sample(f::SphereSDF, point::Point3f) = norm(point - f.origin) - f.radius

struct CubeSDF <: SDF
  origin::Point3f
  length::Scalar
end

sample(f::CubeSDF, point::Point3f) = max(abs.(point - f.origin)...) - f.length

end