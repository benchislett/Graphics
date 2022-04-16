module OBJ

using GeometryBasics: Mesh, coordinates, faces
using FileIO
using ..GeometryTypes

export OBJMeshScene

struct OBJMeshScene <: Scene
  triangles::Vector{Triangle}
end

function OBJMeshScene(filename::String)
  mesh::Mesh = load(filename)

  function facetotri(f)
    pts = coordinates(mesh)[f]
    Triangle(pts[1], pts[2], pts[3])
  end
  tris = map(facetotri, faces(mesh))
  OBJMeshScene(tris)
end

end