include("../src/lib.jl")

using Plots

using .GraphicsCore.Renderer
using .GraphicsCore.OBJ
using .GraphicsCore.Cameras
using .GraphicsCore.GeometryTypes

width = 2048
height = 2048

scene = OBJMeshScene("bunny.obj")
# scene = OBJMeshScene([Triangle([-0.5, 0, 0], [1, 0, 0], [0, 2, 0])])
cam = PerspectiveCamera(width / height, π / 6.3, [-0.3, 1.8, 4], [-0.3, 0.8, 0])

@time img = render(scene, cam, width, height)
img = map(x -> isnan(x) ? RGB(1, 1, 1) : x, img)
plot(img)
png("bunny.png")