#pragma once

#include "helper_math.cuh"
#include "scene.cuh"
#include "trace.cuh"

#include <string>

Scene loadMesh(const std::string &filename);

void writePPM(const std::string &filename, const Image &image);
