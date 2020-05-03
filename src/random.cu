#include "random.cuh"

int RNG::generate_int(int low, int high) const {
  std::uniform_int_distribution<int> distribution{low, high};

  return distribution(*twister);
}

float RNG::generate_float(float low, float high) const {
  std::uniform_real_distribution<float> distribution{low, high};

  return distribution(*twister);
}

__global__ void random_init(int xmax, curandState *state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= xmax) return;

  curand_init(1984, i, 0, &state[i]);
}

DeviceRNG::DeviceRNG(int n_threads) {
  cudaMalloc((void **)&state, n_threads * sizeof(curandState));
  random_init<<<n_threads / 64 + 1, 64>>>(n_threads, state);
  cudaDeviceSynchronize();
}

__device__ float LocalDeviceRNG::generate() const {
  return curand_uniform(local_state);
}

__device__ int LocalDeviceRNG::generate_int(int low, int high) const {
  float r = 1.f - generate();
  return low + (int)(r * (high - low + 1));
}

__device__ float LocalDeviceRNG::generate_float(float low, float high) const {
  float r = generate();
  return low + (r * (high - low));
}
