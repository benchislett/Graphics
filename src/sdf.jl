module SDFs

using LinearAlgebra
using ..GeometryTypes

export SDF, SphereSDF, CubeSDF, DifferenceSDF
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

struct DifferenceSDF <: SDF
  a::SDF
  b::SDF
end

sample(f::DifferenceSDF, point::Point3f) = max(sample(f.a, point), -sample(f.b, point))

end