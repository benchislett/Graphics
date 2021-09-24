#pragma once

#include <algorithm>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

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
using Vector = std::vector<T, UnifiedMemoryAllocator<T>>;

template <typename T>
struct DeviceArray {
  T* data;
  size_t size;

  DeviceArray(Vector<T>& src) : data(src.data()), size(src.size()) {}

  __host__ __device__ T& operator[](size_t i) const {
    if (i >= size)
      throw std::out_of_range();
    return data[i];
  }

  __host__ __device__ T* begin() const {
    return data;
  }

  __host__ __device__ T* end() const {
    return data + size;
  }
};
