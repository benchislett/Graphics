module GeometryCore

using StaticArrays

export Scalar, ScalarInt
export UnitVector3, UnitVector3f
export Vector3, Vector3f, Vector3i
export Vector2, Vector2f, Vector2i
export Point3, Point3f, Point3i
export HitEpsilon

const Scalar = Float32
const ScalarInt = Int32

const UnitVector3{T} = SVector{3,T} where {T<:Number}
const UnitVector3f = UnitVector3{Scalar}

const Vector3{T} = SVector{3,T} where {T<:Number}
const Vector3f = Vector3{Scalar}
const Vector3i = Vector3{ScalarInt}

const Vector2{T} = SVector{2,T} where {T<:Number}
const Vector2f = Vector2{Scalar}
const Vector2i = Vector2{ScalarInt}

const Point3{T} = SVector{3,T} where {T<:Number}
const Point3f = Point3{Scalar}
const Point3i = Point3{ScalarInt}

const HitEpsilon::Scalar = 1e-5

end