#pragma once

#include "cu_error.cuh"

#include <vector>

#define HD  __host__ __device__
#define IHD inline HD

template <typename T>
struct DeviceVector;

template <typename T>
struct HostVector {
  int size;
  int capacity;
  T* data;

  __host__ HostVector() : size(0), capacity(0), data(NULL) {}
  __host__ HostVector(int n) : size(0), capacity(0), data(NULL) {
    resize(n);
  }

  __host__ void realloc(int n) {
    if (capacity >= n) {
      return;
    }

    capacity = n + 1;

    T* new_data = (T*) malloc(capacity * sizeof(T));
    assertNotNull(new_data);
    if (data != NULL) {
      memcpy(new_data, data, capacity * sizeof(T));
      free(data);
    }
    data = new_data;
  }

  __host__ void resize(int n) {
    realloc(n);
    size = capacity;
  }

  __host__ T& operator[](const int i) {
    return data[i];
  }

  __host__ void push_back(const T& item) {
    size++;
    if (size >= capacity) {
      realloc(2 * size);
    }
    data[size - 1] = item;
  }

  __host__ HostVector& operator=(const DeviceVector<T>& device_vector) {
    resize(device_vector.size);
    assertNotNull(data);
    cudaMemcpy(data, device_vector.data, device_vector.size * sizeof(T), cudaMemcpyDeviceToHost);
    printf("Copy device to host\n");
    cudaCheckError();
    return *this;
  }

  __host__ void destroy() {
    if (data != NULL) {
      free(data);
    }
  }
};

template <typename T>
struct DeviceVector {
  int size;
  T* data;

  __host__ DeviceVector() {
    size = 0;
    data = NULL;
  }

  __host__ DeviceVector(int n) {
    size = 0;
    data = NULL;
    resize(n);
  }

  __host__ void resize(int n) {
    if (size >= n) {
      if (n == 0) {
        cudaFree(data);
        cudaCheckError();
        data = NULL;
      }
    } else {
      T* new_data;
      cudaMalloc((void**) &new_data, n * sizeof(T));
      cudaCheckError();
      if (data != NULL) {
        cudaMemcpy(new_data, data, size * sizeof(T), cudaMemcpyDeviceToDevice);
        cudaCheckError();
        cudaFree(data);
        cudaCheckError();
      }
      data = new_data;
    }
    size = n;
  }

  __device__ T& operator[](const int i) {
    return data[i];
  }

  __host__ DeviceVector& operator=(const HostVector<T>& host_vector) {
    resize(host_vector.size);
    assertNotNull(data);
    cudaMemcpy(data, host_vector.data, host_vector.size * sizeof(T), cudaMemcpyHostToDevice);
    cudaCheckError();
    return *this;
  }

  __host__ void destroy() {
    if (data != NULL) {
      cudaFree(data);
      cudaCheckError();
    }
  }
};
