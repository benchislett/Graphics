using Plots, Images, ImageView

using GraphicsCore.Renderer
using GraphicsCore.OBJ
using GraphicsCore.Cameras
using GraphicsCore.GeometryTypes
using GraphicsCore.SDFs

function getscene(sz=2048)
  width = sz
  height = sz

  # SDF sample scene
  scene = SphereSDF([0, 0, 0], 1)
  # scene = DifferenceSDF(CubeSDF([0, 0, 0], 1), SphereSDF([0, 0, 0], 1.25))
  # scene = UnionSDF(scene, SphereSDF([0, 0, 0], 0.8))
  scene = ModuloSDF(scene, [5, 5, 5])
  cam = PerspectiveCamera(width / height, π / 4, [4, 2, 0], [0, 2, 0])

  # Mesh sample scene
  # scene = loadobjmesh("bunny.obj")
  # cam = PerspectiveCamera(width / height, π / 4, [2, 3, 2], [0, 1, 0])

  (scene, cam, width, height)
end

scene = getscene()

@time img = render(scene...)
# save("output.png", img)
# imshow(img)
plot(img)