#pragma once

#include "scene.cuh"
#include "render.cuh"

#include <string>

void load_obj(std::string fname, Scene *scene);

void write_ppm(const std::string &fname, const Image &im);
