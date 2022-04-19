module OBJ

using GeometryBasics: Mesh, coordinates, faces, normals
using FileIO
using ..GeometryTypes

export loadobjmesh

function loadobjmesh(filename::String)
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
    return TriangleArray(tris)
  else
    trinormals = map(facetonormals, faces(mesh))
    return TriangleArray(tris, trinormals)
  end
end

end