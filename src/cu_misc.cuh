#pragma once

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define cudaCheckError()                                                                        \
  {                                                                                             \
    cudaError_t e = cudaGetLastError();                                                         \
    if (e != cudaSuccess) {                                                                     \
      fprintf(stderr, "Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(e);                                                                                  \
    }                                                                                           \
  }

#define assertNotNull(p)                                                             \
  {                                                                                  \
    if ((p) == NULL) {                                                               \
      fprintf(stderr, "Pointer null assertion failed %s:%d:\n", __FILE__, __LINE__); \
      exit(1);                                                                       \
    }                                                                                \
  }

#define HD  __host__ __device__
#define IHD inline HD

template <typename T>
struct HostVector {
  int size;
  T* data;

  __host__ HostVector(int n) {
    size = n;
    data = (T*) malloc(n * sizeof(T));
    assertNotNull(data);
  }

  __host__ T& operator[](int i) {
    return data[i];
  }

  __host__ ~HostVector() {
    free(data);
  }
};

template <typename T>
struct DeviceVector {
  int size;
  T* data;

  __host__ DeviceVector(int n) {
    size = n;
    cudaMalloc((void**) &data, n * sizeof(T));
    cudaCheckError();
  }

  __device__ T& operator[](int i) {
    return data[i];
  }

  __host__ ~DeviceVector() {
    cudaFree(data);
    cudaCheckError();
  }
};
