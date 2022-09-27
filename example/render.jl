using Plots, Images, ImageView, ImageFiltering

using GraphicsCore.Renderer
using GraphicsCore.OBJ
using GraphicsCore.Cameras
using GraphicsCore.GeometryCore
using GraphicsCore.Materials

function getscene(sz=32, t=0)
  width = sz
  height = sz

  # Mesh sample scene
  scene = loadobjmesh("cornell_box.obj")
  cam = PerspectiveCamera(width / height, π / 4, [0, 2.5, 7.5], [0, 2.5, -5])
  materials = [DefaultMaterial() for _ in scene]
  light = 1
  Scene(scene, materials, light, cam, width, height)
end

# t = 0
function makeanim(sz=1024, int=0:0.01:2π)
  @animate for t = int
    scene = getscene(sz, t)
    img = render!(scene)
    # save("output.png", img)
    plot(img; dpi=600, framestyle=:none, background=RGB(0, 0, 0))
  end
end

function makeimage(sz=1024)
  scene = getscene(sz)
  @time render!(scene, NormalsIntegrator())
end

# mp4(makeanim(512, 0:0.005:2π), "./sdf_ao_anim.mp4", fps=60)
# scene = getscene(1024)
# @profview render!(scene)
img = makeimage(128)