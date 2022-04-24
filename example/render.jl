using Plots, Images, ImageView

using GraphicsCore.Renderer
using GraphicsCore.OBJ
using GraphicsCore.Cameras
using GraphicsCore.GeometryTypes
using GraphicsCore.SDFs

function getscene(sz=32, t=0)
  width = sz
  height = sz

  # SDF sample scene
  # scene = SphereSDF([0, 0, 0], 1)
  scene = DifferenceSDF(CubeSDF([0, 0, 0], 1), SphereSDF([0, 0, 0], 1.25))
  scene = UnionSDF(scene, SphereSDF([0, 0, 0], 0.8))
  scene = IntersectionSDF(scene, SphereSDF([0, 0, 0], 1.3))
  # scene = ModuloSDF(scene, [5, 5, 5])
  # cam = PerspectiveCamera(width / height, π / 4, [4, 4, 4], [0, 0, 0])
  cam = PerspectiveCamera(width / height, π / 4, 1.5 .* [2cos(3t), sin(2t)^2 - 2cos(t), 3sin(3t)], [0, 0, 0])

  # Mesh sample scene
  # scene = loadobjmesh("bunny.obj")
  # cam = PerspectiveCamera(width / height, π / 4, [2, 3, 2], [0, 1, 0])

  (scene, cam, width, height)
end

# t = 0
function makeanim(sz=1024, int=0:0.01:2π)
  @animate for t = int
    scene = getscene(sz, t)
    img = render(scene...)
    # save("output.png", img)
    plot(img; dpi=600, framestyle=:none, background=RGB(0, 0, 0))
  end
end

function makeimage(sz=1024)
  scene = getscene(sz)
  render(scene...)
end

# img = makeimage(512)
gif(makeanim(512, 0:0.01:2π), fps=30)