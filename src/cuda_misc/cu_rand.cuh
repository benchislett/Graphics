#pragma once

#include "cu_misc.cuh"

#include <curand_kernel.h>


struct DeviceRNG {
  curandState* state;

  __device__ float uniform(unsigned int tid) {
    curandState* localState = &state[tid];
    return curand_uniform(localState);
  }
};

__global__ void rng_init_kernel(unsigned int xmax, curandState* state) {
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= xmax)
    return;

  curand_init(1984, i, 0, &state[i]);
}

__host__ DeviceRNG init_rng(unsigned int n_threads) {
  DeviceRNG rng;
  cudaMalloc((void**) &rng.state, n_threads * sizeof(curandState));
  rng_init_kernel<<<n_threads / 64 + 1, 64>>>(n_threads, rng.state);
  cudaDeviceSynchronize();
  cudaCheckError();
  return rng;
}