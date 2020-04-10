#include "io.cuh"

#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>


Tri *load_tris_obj(const std::string &fname, int *n) {
  float x,y,z;
  int32_t a,b,c,d,e,f;
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
    } else if (line[0] == 'f' && line.find("/") != std::string::npos) {
      if (line.find("//") != std::string::npos) {
        sscanf(line.c_str(), "f %d//%d %d//%d %d//%d", &a, &b, &c, &d, &e, &f);
      } else {
        sscanf(line.c_str(), "f %d/%*d/%d %d/%*d/%d %d/%*d/%d", &a, &b, &c, &d, &e, &f);
      }
      a = (a < 0 ? verts.size() + a : a - 1);
      b = (b < 0 ? normals.size() + b : b - 1);
      c = (c < 0 ? verts.size() + c : c - 1);
      d = (d < 0 ? normals.size() + d : d - 1);
      e = (e < 0 ? verts.size() + e : e - 1);
      f = (f < 0 ? normals.size() + f : f - 1);
      tris.push_back(Tri(verts[a], verts[c], verts[e], normals[b], normals[d], normals[f]));
    } else if (line[0] == 'f') {
      sscanf(line.c_str(), "f %d %d %d", &a, &b, &c);
      a = (a < 0 ? verts.size() + a : a - 1);
      b = (b < 0 ? verts.size() + b : b - 1);
      c = (c < 0 ? verts.size() + c : c - 1);
      tris.push_back(Tri(verts[a], verts[b], verts[c]));
    }
  }

  Tri *tri_arr = (Tri *)malloc(tris.size() * sizeof(Tri));
  for (int i = 0; i < tris.size(); i++) {
    tri_arr[i] = tris[i];
  }

  *n = tris.size();
  return tri_arr;
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
