include("../src/GraphicsCore.jl")

using Plots

using .GraphicsCore.Renderer
using .GraphicsCore.OBJ
using .GraphicsCore.Cameras
using .GraphicsCore.GeometryTypes
using .GraphicsCore.SDFs

# scene = DifferenceSDF(CubeSDF([0, 0, 0], 1), SphereSDF([0, 0, 0], 1.2))

function getscene()
  width = 32
  height = 32
  scene = loadobjmesh("bunny.obj")
  cam = PerspectiveCamera(width / height, π / 4, [2, 3, 2], [0, 1, 0])

  (scene, cam, width, height)
end
