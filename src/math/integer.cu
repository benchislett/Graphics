#include "integer.cuh"

size_t next_power_2(size_t n) {
  size_t pwr = 1;
  while (pwr < n)
    pwr *= 2;

  return pwr;
}
