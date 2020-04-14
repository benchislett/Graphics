#pragma once

#include "primitive.cuh"
#include "render.cuh"

#include <string>

Primitive *load_obj(const std::string &fname, int *n);

void write_ppm(const std::string &fname, const Image &im);
