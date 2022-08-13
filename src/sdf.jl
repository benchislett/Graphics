module SDFs

using LinearAlgebra
using ..GeometryTypes

export SDF
export SphereSDF, CubeSDF, DifferenceSDF, IntersectionSDF, UnionSDF, ModuloSDF
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

struct DifferenceSDF{T1<:SDF,T2<:SDF} <: SDF
  a::T1
  b::T2
end

sample(f::DifferenceSDF, point::Point3f) = max(sample(f.a, point), -sample(f.b, point))

struct IntersectionSDF{T1<:SDF,T2<:SDF} <: SDF
  a::T1
  b::T2
end

sample(f::IntersectionSDF, point::Point3f) = max(sample(f.a, point), sample(f.b, point))

struct UnionSDF{T1<:SDF,T2<:SDF} <: SDF
  a::T1
  b::T2
end

sample(f::UnionSDF, point::Point3f) = min(sample(f.a, point), sample(f.b, point))

struct ModuloSDF{T<:SDF} <: SDF
  a::T
  period::Vector3f
end

function sample(f::ModuloSDF, point::Point3f)
  q = mod.(point .+ 0.5f0 .* f.period, f.period) .- 0.5f0 .* f.period
  sample(f.a, q)
end

end