#include "io.cuh"
#include "bvh.cuh"

#include <vector>
#include <iostream>
#include <fstream>

#include <cstdio>

std::string ltrim(const std::string &s) {
  size_t start = s.find_first_not_of(" \n\r\t");
  return (start == std::string::npos) ? "" : s.substr(start);
}

std::string rtrim(const std::string &s) {
  size_t end = s.find_last_not_of(" \n\r\t");
  return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}

std::string trim(const std::string &s) {
  return rtrim(ltrim(s));
}

inline bool startsWith(const std::string &tested, const std::string &test) {
  return tested.rfind(test, 0) == 0;
}

Scene loadMesh(const std::string &filename) {
  int i;
  int3 tmp1, tmp2;
  float3 tmp;
  std::vector<float3> vertices;
  std::vector<float3> normals;
  std::vector<Primitive> primitives;
  std::vector<int> lights;
  float emittance = 0.f; // temporary, until material parsing is added
  
  std::ifstream input(filename);

  for (std::string line; std::getline(input, line);) {
    line = trim(line);
    
    if (line[0] == '#' || line[0] == 'g' || line[0] == 'o' || line[0] == 's' || line[0] == 'l' || line == "" || line == "\n") {
      continue;
    } else if (startsWith(line, "usemtl")) {
      if (line.find("lum_bright") != std::string::npos) {
        emittance = 4.0f;
      } else {
        emittance = 0.f;
      }
    } else if (startsWith(line, "vn")) {
      sscanf(line.c_str(), "vn %f %f %f", &tmp.x, &tmp.y, &tmp.z);
      normals.push_back(tmp);
    } else if (startsWith(line, "vt")) {
      continue; // texture coordinates
    } else if (startsWith(line, "v ")) {
      sscanf(line.c_str(), "v %f %f %f", &tmp.x, &tmp.y, &tmp.z);
      vertices.push_back(tmp);
    } else if (startsWith(line, "f ")) {
      sscanf(line.c_str(), "f %u//%u %u//%u %u//%u", &tmp1.x, &tmp2.x, &tmp1.y, &tmp2.y, &tmp1.z, &tmp2.z);
      tmp1.x = tmp1.x < 0 ? vertices.size() + tmp1.x : tmp1.x - 1;
      tmp1.y = tmp1.y < 0 ? vertices.size() + tmp1.y : tmp1.y - 1;
      tmp1.z = tmp1.z < 0 ? vertices.size() + tmp1.z : tmp1.z - 1;
      tmp2.x = tmp2.x < 0 ? normals.size() + tmp2.x : tmp2.x - 1;
      tmp2.y = tmp2.y < 0 ? normals.size() + tmp2.y : tmp2.y - 1;
      tmp2.z = tmp2.z < 0 ? normals.size() + tmp2.z : tmp2.z - 1;
      Tri t = {vertices[tmp1.x], vertices[tmp1.y], vertices[tmp1.z], normals[tmp2.x], normals[tmp2.y], normals[tmp2.z]};
      AABB bound = {fminf(fminf(t.a, t.b), t.c), fmaxf(fmaxf(t.a, t.b), t.c)};
      primitives.emplace_back(t, bound, emittance);
      if (emittance != 0.f) lights.push_back(primitives.size() - 1);
    }
  }

  int primitiveCount = primitives.size();
  Primitive *primitives_arr = (Primitive *)malloc(primitiveCount * sizeof(Primitive));
  for (i = 0; i < primitiveCount; i++) primitives_arr[i] = primitives[i];

  int lightCount = lights.size();
  int *lights_arr = (int *)malloc(lightCount * sizeof(int));
  for (i = 0; i < lightCount; i++) lights_arr[i] = lights[i];

  BVH bvh(primitiveCount, primitives_arr);

  return {primitiveCount, primitives_arr, lightCount, lights_arr, bvh};
}

void writePPM(const std::string &fname, const Image &image) {
  std::ofstream output(fname);

  output << "P3\n" << image.width << ' ' << image.height << "\n255\n";

  float3 rgb;
  int r, g, b;
  for (int j = image.height - 1; j >= 0; j--) {
    for (int i = 0; i < image.width; i++) {
      rgb = image.data[j * image.width + i];
      rgb.x = (std::isnan(rgb.x)) ? 0.f : ((rgb.x > 1.f) ? 1.f : rgb.x);
      rgb.y = (std::isnan(rgb.y)) ? 0.f : ((rgb.y > 1.f) ? 1.f : rgb.y);
      rgb.z = (std::isnan(rgb.z)) ? 0.f : ((rgb.z > 1.f) ? 1.f : rgb.z);
      r = (int)(255.999 * sqrtf(rgb.x));
      g = (int)(255.999 * sqrtf(rgb.y));
      b = (int)(255.999 * sqrtf(rgb.z));
      output << r << ' ' << g << ' ' << b << '\n';
    }
  }
}
