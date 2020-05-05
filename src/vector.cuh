#pragma once

#include "cuda.cuh"

template<typename T>
struct Vector {
  T *data;
  int n;
  bool is_host;

  Vector() : data(NULL), n(0), is_host(true) {}
  Vector(int n) : n(n), is_host(true) { data = (T *)calloc(n, sizeof(T)); }
  Vector(T *data, int n) : data(data), n(n), is_host(true) {}

  __host__ __device__ T& operator[](int i) const { return data[i]; }
  __host__ __device__ int size() const { return n; }

  void to_device() {
    if (!is_host) return;
    T *device_data;
    cudaMalloc((void **)&device_data, n * sizeof(T));
    cudaMemcpy(device_data, data, n * sizeof(T), cudaMemcpyHostToDevice);
    free(data);
    data = device_data;
    is_host = false;
  }

  void to_host() {
    if (is_host) return;
    T *host_data = (T *)malloc(n * sizeof(T));
    cudaMemcpy(host_data, data, n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(data);
    data = host_data;
    is_host = true;
  }
};
