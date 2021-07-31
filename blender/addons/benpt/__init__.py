import bpy
import bgl
import benptpy


bl_info = {
    "name": "benpt",
    "author": "Benjamin Chislett",
    "version": (2, 0, 0),
    "blender": (2, 93, 0),
    "description": "Benpt blender integration",
    "location": "Rendering > Render Engine",
    "warning": "",
    "category": "Render",
}


class CustomRenderEngine(bpy.types.RenderEngine):
    # These three members are used by blender to set up the
    # RenderEngine; define its internal name, visible name and capabilities.
    bl_idname = "CUSTOM"
    bl_label = "Custom"
    bl_use_preview = True

    # Init is called whenever a new render engine instance is created. Multiple
    # instances may exist at the same time, for example for a viewport and final
    # render.
    def __init__(self):
        self.scene_data = None
        self.draw_data = None

    # When the render engine instance is destroy, this is called. Clean up any
    # render engine data here, for example stopping running render threads.
    def __del__(self):
        pass

    # This is the method called by Blender for both final renders (F12) and
    # small preview for materials, world and lights.
    def render(self, depsgraph):
        scene = depsgraph.scene
        scale = scene.render.resolution_percentage / 100.0
        self.size_x = int(scene.render.resolution_x * scale)
        self.size_y = int(scene.render.resolution_y * scale)

        # Fill the render result with a flat color. The framebuffer is
        # defined as a list of pixels, each pixel itself being a list of
        # R,G,B,A values.
        if self.is_preview:
            color = [0.1, 0.0, 0.1, 1.0]
        else:
            color = [0.2, 0.0, 0.8, 1.0]

        tris = [[[2, -1, -1], [2, 1, -1], [2, 0, 1]]]
        rect = benptpy.render(tris, self.size_x, self.size_y)
        print(tuple(map(int, benptpy.__version__.split("."))))
        # Here we write the pixel values to the RenderResult
        result = self.begin_result(0, 0, self.size_x, self.size_y)
        layer = result.layers[0].passes["Combined"]
        layer.rect = rect
        self.end_result(result)


# RenderEngines also need to tell UI Panels that they are compatible with.
# We recommend to enable all panels marked as BLENDER_RENDER, and then
# exclude any panels that are replaced by custom panels registered by the
# render engine, or that are not supported.
def get_panels():
    exclude_panels = {
        "VIEWLAYER_PT_filter",
        "VIEWLAYER_PT_layer_passes",
    }

    panels = []
    for panel in bpy.types.Panel.__subclasses__():
        if (
            hasattr(panel, "COMPAT_ENGINES")
            and "BLENDER_RENDER" in panel.COMPAT_ENGINES
        ):
            if panel.__name__ not in exclude_panels:
                panels.append(panel)

    return panels


def register():
    # Register the RenderEngine
    bpy.utils.register_class(CustomRenderEngine)

    for panel in get_panels():
        panel.COMPAT_ENGINES.add("CUSTOM")


def unregister():
    bpy.utils.unregister_class(CustomRenderEngine)

    for panel in get_panels():
        if "CUSTOM" in panel.COMPAT_ENGINES:
            panel.COMPAT_ENGINES.remove("CUSTOM")


if __name__ == "__main__":
    register()
