module Sampling

using LinearAlgebra
using Statistics

using ..GeometryTypes

export sample_sphere, sample_oriented_hemisphere

function sample_sphere(u::Scalar, v::Scalar)::Vector3f
  z = 1 - 2u
  r = sqrt(max(0, 1 - z^2))
  Φ = 2π * v
  normalize(Vector3f(r * cos(Φ), r * sin(Φ), z))

  # θ0 = 2π * u
  # θ1 = acos(1 - 2v)
  # x = sin(θ0)sin(θ1)
  # y = cos(θ0)sin(θ1)
  # z = cos(θ1)

  # Vector3f(x, y, z)
end

function sample_oriented_hemisphere(n, u::Scalar, v::Scalar)::Vector3f
  w = sample_sphere(u, v)
  if dot(w, n) < 0
    w = -w
  end

  w
end

end