using Revise

include("../src/lib.jl")

using Plots

using .GraphicsCore.Renderer
using .GraphicsCore.OBJ
using .GraphicsCore.Cameras

scene = OBJMeshScene("bunny.obj")
cam = PerspectiveCamera(1, π / 4, [0.3, 0.3, 0.3], [0, 0, 0])

img = render(scene, cam, 128, 128)
plot(img)