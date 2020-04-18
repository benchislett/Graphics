#include "primitive.cuh"

Primitive::Primitive(const Tri &t) : t(t) {
  bsdf = new BSDF(new Lambertian(Vec3(1.f, 1.f, 1.f)));
}
