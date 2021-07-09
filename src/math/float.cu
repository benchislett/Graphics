#include "float.cuh"

__host__ __device__ int sign(float f) {
  if (f == 0) {
    return 0;
  } else if (f > 0) {
    return 1;
  } else {
    return -1;
  }
}

__host__ __device__ float fclamp(float x, float a, float b) {
  if (x <= a) {
    return a;
  } else if (x >= b) {
    return b;
  } else {
    return x;
  }
}
