#pragma once

#include <cuda_runtime.h>

__host__ __device__ int sign(float f);
__host__ __device__ float fclamp(float x, float a, float b);
