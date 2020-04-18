#pragma once

#include "scene.cuh"
#include "render.cuh"

#include <string>

Scene load_obj(std::string fname);

void write_ppm(const std::string &fname, const Image &im);
