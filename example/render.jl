include("../src/lib.jl")

using Plots

using .GraphicsCore.Renderer
using .GraphicsCore.OBJ
using .GraphicsCore.Cameras
using .GraphicsCore.GeometryTypes
using .GraphicsCore.SDFs

width = 128
height = 128

scene = CubeSDF([0, 0, 0], 1)
cam = PerspectiveCamera(width / height, π / 4, [3, 3, 3], [0, 0, 0])

@time img = render(scene, cam, width, height)
img = map(x -> isnan(x) ? RGB(1, 1, 1) : x, img)
plot(img)
