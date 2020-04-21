#include "io.cuh"

#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <map>

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

void load_material(std::map<std::string, BSDF> &materials, const std::string &name, const Vec3 &Kd, const Vec3 &Ke) {
  if (name != "") {
    if (Ke.e[0] == 0.f && Ke.e[1] == 0.f && Ke.e[2] == 0.f) {
      materials[name] = BSDF(new Lambertian(Kd));
    } else {
      materials[name] = BSDF(new AreaLight(Kd, Ke));
    }
  }
}

void load_materials(std::string &fname, std::map<std::string, BSDF> &materials) {
  std::string current_name = "";
  Vec3 Kd;
  Vec3 Ke;

  std::ifstream input(fname);
  std::string line;
  for (; std::getline(input, line); ) {
    line = trim(line);
    if (line.find("newmtl") != std::string::npos) {
      load_material(materials, current_name, Kd, Ke);
      current_name = line.substr(7);
      Kd = {0.f, 0.f, 0.f};
      Ke = {0.f, 0.f, 0.f};
    } else if (line[0] == 'K' && line[1] == 'd') {
      sscanf(line.c_str(), "Kd %f %f %f %*s", Kd.e, Kd.e + 1, Kd.e + 2);
    } else if (line[0] == 'K' && line[1] == 'e') {
      sscanf(line.c_str(), "Ke %f %f %f %*s", Ke.e, Ke.e + 1, Ke.e + 2);
    }
  }
  load_material(materials, current_name, Kd, Ke);
}

Scene load_obj(std::string fname) {
  Camera cam;

  float x,y,z;
  int a,b,c,d,e,f;
  std::ifstream input(fname);
  std::map<std::string, BSDF> materials;
  materials[""] = BSDF(new Lambertian(Vec3(1.f, 1.f, 1.f)));
  fname.replace(fname.end() - 3, fname.end(), "mtl");
  std::string current_name = "";
  load_materials(fname, materials);

  int n_mats = materials.size();
  BSDF *mats = (BSDF *)malloc(n_mats * sizeof(BSDF));
  int i = 0;
  for (auto it = materials.begin(); it != materials.end(); it++, i++) mats[i] = it->second; 

  std::vector<Vec3> verts;
  std::vector<Vec3> normals;
  std::vector<Primitive> prims;
  BSDF *bsdf;
  int idx;
  int n_lights = 0;

  for (std::string line; std::getline(input, line); ) {
    line = trim(line);
    if (line[0] == 'v' && line[1] == ' ') {
      sscanf(line.c_str(), "v %f %f %f", &x, &y, &z);
      verts.emplace_back(x, y, z);
    } else if (line[0] == 'v' && line[1] == 'n') {
      sscanf(line.c_str(), "vn %f %f %f", &x, &y, &z);
      normals.emplace_back(x, y, z);
    } else if (line.find("usemtl") != std::string::npos) {
      current_name = line.substr(7);
    } else if (line[0] == 'f') {
      idx = std::distance(materials.begin(), materials.find(current_name));
      if (idx == materials.size()) printf("No material with name %s\n", current_name.c_str());
      bsdf = mats + idx;
      if (bsdf->is_light()) n_lights++;
      if (line.find("/") != std::string::npos) {
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
        prims.emplace_back(Tri(verts[a], verts[c], verts[e], normals[b], normals[d], normals[f]), bsdf);
      } else if (line[0] == 'f') {
        sscanf(line.c_str(), "f %d %d %d", &a, &b, &c);
        a = (a < 0) ? verts.size() + a : a - 1;
        b = (b < 0) ? verts.size() + b : b - 1;
        c = (c < 0) ? verts.size() + c : c - 1;
        prims.emplace_back(Tri(verts[a], verts[b], verts[c]), bsdf);
      }
    } else if (line != "" && line != "\n") {
      printf("Unrecognized line %s\n", line.c_str());
    }
  }
  Primitive *prim_arr = (Primitive *)malloc(prims.size() * sizeof(Primitive));
  Primitive **lights = (Primitive **)malloc(n_lights * sizeof(Primitive *));

  int light = 0;
  for (int i = 0; i < prims.size(); i++) {
    prim_arr[i] = prims[i];
    if (prim_arr[i].bsdf->is_light()) lights[light++] = prim_arr + i;
  }

  BVH bvh = build_bvh(prim_arr, prims.size());
  return {cam, bvh, lights, n_lights, mats, n_mats};
}

void write_ppm(const std::string &fname, const Image &im) {
  std::ofstream output(fname);

  output << "P3\n" << im.width << ' ' << im.height << "\n255\n";

  Vec3 rgb;
  int r, g, b;
  for (int j = im.height - 1; j >= 0; j--) {
    for (int i = 0; i < im.width; i++) {
      rgb = im.film[j * im.width + i];
      rgb.e[0] = (std::isnan(rgb.e[0])) ? 0.f : ((rgb.e[0] > 1.f) ? 1.f : rgb.e[0]);
      rgb.e[1] = (std::isnan(rgb.e[1])) ? 0.f : ((rgb.e[1] > 1.f) ? 1.f : rgb.e[1]);
      rgb.e[2] = (std::isnan(rgb.e[2])) ? 0.f : ((rgb.e[2] > 1.f) ? 1.f : rgb.e[2]);
      r = (int)(255.999 * sqrtf(rgb.e[0]));
      g = (int)(255.999 * sqrtf(rgb.e[1]));
      b = (int)(255.999 * sqrtf(rgb.e[2]));
      output << r << ' ' << g << ' ' << b << '\n';
    }
  }
}
