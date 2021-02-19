#pragma once

#include "cu_error.cuh"

#define HD  __host__ __device__
#define IHD inline HD

template <typename T>
struct DeviceVector;

template <typename T>
struct HostVector {
  int size;
  T* data;

  __host__ HostVector(int n) {
    size = n;
    data = (T*) malloc(n * sizeof(T));
    assertNotNull(data);
  }

  __host__ T& operator[](const int i) {
    return data[i];
  }

  __host__ HostVector& operator=(const DeviceVector<T>& device_vector) {
    cudaMemcpy(data, device_vector.data, size * sizeof(T), cudaMemcpyDeviceToHost);
    printf("Copy device to host\n");
    cudaCheckError();
    return *this;
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

  __device__ T& operator[](const int i) {
    return data[i];
  }

  __host__ DeviceVector& operator=(const HostVector<T>& host_vector) {
    cudaMemcpy(data, host_vector.data, size * sizeof(T), cudaMemcpyHostToDevice);
    cudaCheckError();
    return *this;
  }

  __host__ ~DeviceVector() {
    // cudaFree(data);
    cudaCheckError();
  }
};
