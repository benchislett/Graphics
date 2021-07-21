#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#include "render.cuh"


PYBIND11_MODULE(benptpy_core, m) {

  m.def(
      "render",
      [](const py::int_ w, const py::int_ h) {
        int width  = w;
        int height = h;
        std::vector<std::array<float, 4>> out(width * height);
        Sphere s({2, 0, 0}, 0.5);
        Camera cam(M_PI / 4.0, 1.0, {-1, 0, 0}, {1, 0, 0});
        Image img = render_normals(s, cam, width, height);
        for (int i = 0; i < width * height; i++) {
          out[i] = {img[i].x, img[i].y, img[i].z, 1.0};
        }
        img.destroy();
        return out;
      },
      "Render");

  m.attr("__version__") = "2.0.1";
}
