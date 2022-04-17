module OBJ

using GeometryBasics: Mesh, coordinates, faces, normals
using FileIO
using ..GeometryTypes

export OBJMeshScene

struct OBJMeshScene <: Hittable
  triangles::Vector{Triangle}
  normals::Vector{TriangleNormals}
end

function OBJMeshScene(tris::Vector{Triangle})
  trinormals = map(TriangleNormals, tris)
  OBJMeshScene(tris, trinormals)
end

function OBJMeshScene(filename::String)
  mesh::Mesh = load(filename)

  function facetotri(f)
    pts = coordinates(mesh)[f]
    Triangle(pts[1], pts[2], pts[3])
  end

  function facetonormals(f)
    pts = normals(mesh)[f]
    TriangleNormals(pts[1], pts[2], pts[3])
  end

  tris = map(facetotri, faces(mesh))

  if isnothing(normals(mesh))
    trinormals = map(TriangleNormals, tris)
  else
    trinormals = map(facetonormals, faces(mesh))
  end

  OBJMeshScene(tris, trinormals)
end

end