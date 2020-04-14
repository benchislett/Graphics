#include "primitive.cuh"

Primitive::Primitive(const Tri &t) : t(t) {
  bsdf = BSDF(new OrenNayar(Vec3(1.f, 1.f, 1.f), 0.f));
}
