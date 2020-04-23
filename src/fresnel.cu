#include "fresnel.cuh"

float Fresnel::evaluate(float costhetai) const {
  float R0 = (eta1 - eta2) / (eta1 + eta2);
  R0 = R0 * R0;
  return R0 + (1.f - R0) * (1.f - (costhetai * costhetai) * (costhetai * costhetai) * costhetai);
}
