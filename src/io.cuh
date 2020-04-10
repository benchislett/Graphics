#pragma once

#include "render.cuh"

#include <string>

Tri *load_tris_obj(const std::string &fname, int *n);

void write_tris_ppm(const std::string &fname, const Image &im);
