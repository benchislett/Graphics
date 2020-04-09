#include "io.cuh"

#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>


void load_tris_obj(const std::string &fname, Scene *scene) {
  float x,y,z;
  uint32_t a,b,c,d,e,f;
  std::ifstream input(fname);
  std::vector<Vec3> verts;
  std::vector<Vec3> normals;
  std::vector<Tri> tris;

  for (std::string line; std::getline(input, line); ) {
    if (line[0] == 'v' && line[1] == ' ') {
      sscanf(line.c_str(), "v %f %f %f", &x, &y, &z);
      verts.emplace_back(x, y, z);
    } else if (line[0] == 'v' && line[1] == 'n') {
      sscanf(line.c_str(), "vn %f %f %f", &x, &y, &z);
      normals.emplace_back(x, y, z);
    } else if (line[0] == 'f') {
      if (line.find("//") == std::string::npos) {
        sscanf(line.c_str(), "f %u/%*u/%u %u/%*u/%u %u/%*u/%u", &a, &b, &c, &d, &e, &f);
      } else {
        sscanf(line.c_str(), "f %u//%u %u//%u %u//%u", &a, &b, &c, &d, &e, &f);
      }
      tris.emplace_back(Tri(verts[a], verts[c], verts[e], normals[b], normals[d], normals[f]));
    }
  }
  Tri *tri_arr = (Tri *)malloc(tris.size() * sizeof(Tri));
  for (int i = 0; i < tris.size(); i++) {
    tri_arr[i] = tris[i];
  }
  scene->n_tris = tris.size();
  scene->tris = tri_arr;
}

void write_tris_ppm(const std::string &fname, const Image &im) {
  std::ofstream output(fname);

  output << "P3\n" << im.width << ' ' << im.height << "\n255\n";

  Vec3 rgb;
  int r, g, b;
  for (int j = im.height - 1; j >= 0; j--) {
    for (int i = 0; i < im.width; i++) {
      rgb = im.film[j * im.width + i];
      r = (int)(255.999 * sqrtf(rgb.e[0]));
      g = (int)(255.999 * sqrtf(rgb.e[1]));
      b = (int)(255.999 * sqrtf(rgb.e[2]));
      output << r << ' ' << g << ' ' << b << '\n';
    }
  }
}
