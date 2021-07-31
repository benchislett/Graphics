#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#include "render.cuh"

using py_float3 = std::array<float, 3>;
using py_tri    = std::array<py_float3, 3>;

PYBIND11_MODULE(benptpy_core, m) {

  m.def(
      "render",
      [](std::vector<py_tri> tri_values, const py::int_ w, const py::int_ h) {
        int width  = w;
        int height = h;
        std::vector<std::array<float, 4>> out(width * height);
        std::vector<Triangle> tris;
        for (auto t : tri_values) {
          float3 v[3];
          for (int i = 0; i < 3; i++) {
            v[i].x = t[i][0];
            v[i].y = t[i][1];
            v[i].z = t[i][2];
          }
          tris.emplace_back(v[0], v[1], v[2]);
        }
        TriMesh mesh(tris.data(), tris.size());
        Camera cam(M_PI / 4.0, 1.0, {-1, 0, 0}, {1, 0, 0});
        Image img = render_normals(mesh, cam, width, height);
        for (int i = 0; i < width * height; i++) {
          out[i] = {img[i].x, img[i].y, img[i].z, 1.0};
        }
        img.destroy();
        return out;
      },
      "Render");

  m.attr("__version__") = "2.0.1";
}
