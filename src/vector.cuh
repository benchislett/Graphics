#pragma once

template<typename T>
struct Vector {
  T *data;
  int n;

  Vector() : data(NULL), n(0) {}
  Vector(int n) : n(n) { data = (T *)calloc(n, sizeof(T)); }
  Vector(T *data, int n) : data(data), n(n) {}

  T& operator[](int i) const { return data[i]; }
  int size() const { return n; }
};
