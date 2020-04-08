#pragma once

#include "render.cuh"

#include <string>

void load_tris_obj(const std::string &fname, Scene *scene);

void write_tris_ppm(const std::string &fname, const Image &im);
