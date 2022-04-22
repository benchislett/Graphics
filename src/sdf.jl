module SDFs

using LinearAlgebra
using ..GeometryTypes

export SDF
export SphereSDF, CubeSDF, DifferenceSDF, IntersectSDF, UnionSDF, ModuloSDF
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

struct IntersectSDF <: SDF
  a::SDF
  b::SDF
end

sample(f::IntersectSDF, point::Point3f) = max(sample(f.a, point), sample(f.b, point))

struct UnionSDF <: SDF
  a::SDF
  b::SDF
end

sample(f::UnionSDF, point::Point3f) = min(sample(f.a, point), sample(f.b, point))

struct ModuloSDF <: SDF
  a::SDF
  period::Vector3f
end

function sample(f::ModuloSDF, point::Point3f)
  q = mod.(point .+ 0.5f0 .* f.period, f.period) .- 0.5f0 .* f.period
  sample(f.a, q)
end

end