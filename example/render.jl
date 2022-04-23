using Plots, Images, ImageView

using GraphicsCore.Renderer
using GraphicsCore.OBJ
using GraphicsCore.Cameras
using GraphicsCore.GeometryTypes
using GraphicsCore.SDFs

function getscene(sz=64, t=0)
  width = sz
  height = sz

  # SDF sample scene
  # scene = SphereSDF([0, 0, 0], 1)
  scene = DifferenceSDF(CubeSDF([0, 0, 0], 1), SphereSDF([0, 0, 0], 1.25))
  scene = UnionSDF(scene, SphereSDF([0, 0, 0], 0.8))
  scene = IntersectSDF(scene, SphereSDF([0, 0, 0], 1.3))
  # scene = ModuloSDF(scene, [5, 5, 5])
  cam = PerspectiveCamera(width / height, π / 4, [4 * cos(t), 4 - 2 * cos(t), 4 * sin(t)], [0, 0, 0])

  # Mesh sample scene
  # scene = loadobjmesh("bunny.obj")
  # cam = PerspectiveCamera(width / height, π / 4, [2, 3, 2], [0, 1, 0])

  (scene, cam, width, height)
end

t = 0
# anim = @animate for t = 0:0.1:2π
scene = getscene(32, t)
img = render(scene...)
# save("output.png", img)
# plot(img)
# end

# gif(anim, fps=8)
# imshow(img)
# plot(img)