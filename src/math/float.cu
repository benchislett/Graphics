#include "float.cuh"

__host__ __device__ float fclamp(float x, float a, float b) {
  if (x <= a) {
    return a;
  } else if (x >= b) {
    return b;
  } else {
    return x;
  }
}
