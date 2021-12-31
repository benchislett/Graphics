#pragma once

#include <chrono>
#include <functional>
#include <iostream>

struct ScopedMicroTimer {
  std::chrono::_V2::system_clock::time_point t0;
  std::function<void(int)> cb;

  __host__ ScopedMicroTimer(std::function<void(int)> callback)
      : t0(std::chrono::high_resolution_clock::now()), cb(callback) {}

  __host__ ~ScopedMicroTimer(void) {
    auto t1     = std::chrono::high_resolution_clock::now();
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    cb(micros);
  }
};
