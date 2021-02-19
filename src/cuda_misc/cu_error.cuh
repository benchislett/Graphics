#pragma once

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <device_launch_parameters.h>

#define cudaCheckError()                                                                        \
  {                                                                                             \
    cudaError_t e = cudaGetLastError();                                                         \
    if (e != cudaSuccess) {                                                                     \
      fprintf(stderr, "Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(e);                                                                                  \
    }                                                                                           \
  }

#define assertNotNull(p)                                                            \
  {                                                                                 \
    if ((p) == NULL) {                                                              \
      fprintf(stderr, "Pointer null assertion failed %s:%d\n", __FILE__, __LINE__); \
      exit(1);                                                                      \
    }                                                                               \
  }

#define curandCheck(curand_call)                                               \
  {                                                                            \
    curandStatus_t ret = (curand_call);                                        \
    if (ret != CURAND_STATUS_SUCCESS) {                                        \
      fprintf(stderr, "Curand failure %s:%d: '%d'\n", __FILE__, __LINE__, ret) \
    }                                                                          \
  }
