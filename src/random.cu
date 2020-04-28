#include "random.cuh"

int RNG::generate_int(int low, int high) const {
  std::uniform_int_distribution<int> distribution{low, high};

  return distribution(*twister);
}

float RNG::generate_float(float low, float high) const {
  std::uniform_real_distribution<float> distribution{low, high};

  return distribution(*twister);
}
