module OBJ

using GeometryBasics: Mesh, coordinates, faces, normals
using FileIO
using LinearAlgebra

using ..GeometryCore
using ..GeometryMeshes

export loadobjmesh

function loadobjmesh(filename)
  mesh = load(filename)

  castindices(idxs) = convert(Point3i, idxs)
  faceindices = map(castindices, faces(mesh))

  castvertices(coords) = convert(Point3f, coords)
  vertexpositions = map(castvertices, coordinates(mesh))

  castnormals(vals) = normalize(convert(UnitVector3f, vals))
  meshnormals = map(castnormals, normals(mesh))

  TriangleMesh(vertexpositions, faceindices, meshnormals)
end

end