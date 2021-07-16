#include <pybind11/pybind11.h>
namespace py = pybind11;
#include "render.cuh"

PYBIND11_MODULE(benptpy_core, m) {
  m.def(
      "render",
      []() {
        return (float3){1, 1, 1};
      },
      "Render");
}
