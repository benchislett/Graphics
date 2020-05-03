#pragma once

#include "cuda.cuh"

#include <random>

struct RNG {
  std::mt19937 *twister;

  RNG() : twister(new std::mt19937(std::random_device{}())) {}
  RNG(unsigned int seed) : twister(new std::mt19937(seed)) {}

  int generate_int(int low, int high) const;
  float generate_float(float low = 0.f, float high = 1.f) const;
};

struct LocalDeviceRNG {
  curandState *local_state;

  LocalDeviceRNG(curandState *s) : local_state(s) {}

  __device__ float generate() const;
  __device__ int generate_int(int low, int high) const;
  __device__ float generate_float(float low, float high) const;
};

struct DeviceRNG {
  curandState *state;

  DeviceRNG(int n_threads);

  __device__ LocalDeviceRNG local(int tid) {
    return LocalDeviceRNG(&state[tid]);
  }
};
