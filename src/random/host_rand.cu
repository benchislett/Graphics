#include "host_rand.cuh"

__host__ int rand_in_range(int low, int high) {
  static std::mt19937 gen(std::random_device{}());

  std::uniform_int_distribution<> distr(low, high);
  return distr(gen);
}