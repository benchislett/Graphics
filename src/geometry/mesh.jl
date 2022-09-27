module GeometryMeshes

import Base: size, getindex, length, IndexStyle, IndexLinear

import StaticArrays.SVector, StaticArrays.@SVector

using ..GeometryCore
using ..GeometryPrimitives

export TriangleMesh

struct TriangleMesh <: AbstractArray{Triangle,1}
  vertexpositions::Vector{Point3f}
  faceindices::Vector{Point3i}
  normals::Vector{UnitVector3f}
end

Base.IndexStyle(::TriangleMesh) = IndexLinear()
Base.length(mesh::TriangleMesh) = length(mesh.faceindices)
Base.size(mesh::TriangleMesh) = tuple(length(mesh))
Base.getindex(mesh::TriangleMesh, i::Int) =
  Triangle(mesh.vertexpositions[mesh.faceindices[i]])

end