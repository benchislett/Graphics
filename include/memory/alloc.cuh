#pragma once

#include "integer.cuh"

#include <algorithm>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <vector>

#define cudaCheckError()                                                               \
  {                                                                                    \
    cudaError_t e = cudaGetLastError();                                                \
    if (e != cudaSuccess) {                                                            \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(0);                                                                         \
    }                                                                                  \
  }

template <class T>
struct UnifiedMemoryAllocator {
  typedef T value_type;

  UnifiedMemoryAllocator() = default;

  template <class U>
  constexpr UnifiedMemoryAllocator(const UnifiedMemoryAllocator<U>&) noexcept {}

  [[nodiscard]] T* allocate(size_t n) {
    if (n > std::numeric_limits<size_t>::max() / sizeof(T))
      throw std::bad_array_new_length();

    T* p;
    cudaMallocManaged(&p, n * sizeof(T));

    if (p) {
      return p;
    } else {
      throw std::bad_alloc();
    }
  }

  void deallocate(T* p, size_t n) noexcept {
    if (n != 0)
      cudaFree(p);
  }
};

template <class T, class U>
bool operator==(const UnifiedMemoryAllocator<T>&, const UnifiedMemoryAllocator<U>&) {
  return true;
}
template <class T, class U>
bool operator!=(const UnifiedMemoryAllocator<T>&, const UnifiedMemoryAllocator<U>&) {
  return false;
}

template <typename T>
struct Vector {
  T* data;
  size_t size;
  size_t capacity;

  Vector() : data(nullptr), size(0), capacity(0) {}
  Vector(size_t n) : size(n), capacity(n) {
    UnifiedMemoryAllocator<T> allocator;
    data = allocator.allocate(n);
  }
  Vector(size_t n, T value) : Vector(n) {
    std::fill(begin(), end(), value);
  }

  __host__ __device__ T* begin() const noexcept {
    return data;
  }

  __host__ __device__ T* end() const noexcept {
    return data + size;
  }

  __host__ __device__ T& operator[](size_t idx) const {
    return data[idx];
  }

  __host__ void reserve(size_t n) {
    if (n <= capacity) {
      return;
    }

    UnifiedMemoryAllocator<T> allocator;
    T* new_data = allocator.allocate(n);
    std::copy(begin(), end(), new_data);
    allocator.deallocate(data, capacity);
    data = new_data;

    capacity = n;
  }

  __host__ void push_back(const T& value) {
    reserve(next_power_2(size + 1));
    data[size++] = value;
  }

  __host__ void release() {
    UnifiedMemoryAllocator<T> allocator;
    allocator.deallocate(data, capacity);
  }
};
