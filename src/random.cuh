#pragma once

#include <random>

struct RNG {
  std::mt19937 *twister;

  RNG() : twister(new std::mt19937(std::random_device{}())) {}
  RNG(unsigned int seed) : twister(new std::mt19937(seed)) {}

  int generate_int(int low, int high) const;
  float generate_float(float low = 0.f, float high = 1.f) const;
};
