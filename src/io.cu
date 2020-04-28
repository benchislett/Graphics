#include "io.cuh"
#include "bxdf.cuh"
#include "fresnel.cuh"
#include "bsdf.cuh"
#include "texture.cuh"

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

void load_material(std::map<std::string, BSDF> &materials, const std::string &name, float Ns, const Vec3 &Kd, const Vec3 &Ks, const Vec3 &Ke, int tex) {
  if (name != "") {
    if (!is_zero(Ke)) {
      materials[name] = BSDF(new AreaLight(Kd, Ke), tex);
    } else if (is_zero(Ks)) {
      materials[name] = BSDF(new Lambertian(Kd), tex);
    } else {
      float roughness = 1.f - sqrtf(Ns) / 30.f;
      materials[name] = BSDF(new Lambertian(Kd), tex, new TorranceSparrow(Ks, new Beckmann(roughness), new Fresnel(1.0f, 1.5f)));
    }
  }
}

void load_materials(const std::string &fname, std::map<std::string, BSDF> &materials, std::vector<Texture> &textures) {
  std::string dir_prefix = (fname.rfind("/") == std::string::npos) ? "" : fname.substr(0, fname.rfind("/") + 1);
  std::string texture_file(200, '\0');
  std::string current_name = "";
  Vec3 Kd(0.f, 0.f, 0.f);
  Vec3 Ks(0.f, 0.f, 0.f);
  Vec3 Ke(0.f, 0.f, 0.f);
  float Ns = 0.f;
  int current_texture = -1;

  std::ifstream input(fname);
  std::string line;
  for (; std::getline(input, line); ) {
    line = trim(line);
    if (line.find("newmtl") != std::string::npos) {
      load_material(materials, current_name, Ns, Kd, Ks, Ke, current_texture);
      current_name = line.substr(7);
      Kd = {0.f, 0.f, 0.f};
      Ks = {0.f, 0.f, 0.f};
      Ke = {0.f, 0.f, 0.f};
      Ns = 0.f;
      current_texture = -1;
    } else if (line[0] == 'K' && line[1] == 'd') {
      sscanf(line.c_str(), "Kd %f %f %f", Kd.e, Kd.e + 1, Kd.e + 2);
    } else if (line[0] == 'K' && line[1] == 's') {
      sscanf(line.c_str(), "Ks %f %f %f", Ks.e, Ks.e + 1, Ks.e + 2);
    } else if (line[0] == 'K' && line[1] == 'e') {
      sscanf(line.c_str(), "Ke %f %f %f", Ke.e, Ke.e + 1, Ke.e + 2);
    } else if (line[0] == 'N' && line[1] == 's') {
      sscanf(line.c_str(), "Ns %f", &Ns);
    } else if (line.compare(0, 6, "map_Kd") == 0) {
      sscanf(line.c_str(), "map_Kd %s", texture_file.c_str());
      texture_file.insert(0, dir_prefix);
      textures.emplace_back(texture_file);
      current_texture = textures.size() - 1;
    }
  }
  load_material(materials, current_name, Ns, Kd, Ks, Ke, current_texture);
}

void load_vertex(const std::string &line, std::vector<Vec3> &verts) {
  float x, y, z;
  sscanf(line.c_str(), "v %f %f %f", &x, &y, &z);
  verts.emplace_back(x, y, z);
}

void load_vertex_tex(const std::string &line, std::vector<Vec3> &tex_coords) {
  float u, v;
  sscanf(line.c_str(), "vt %f %f", &u, &v);
  tex_coords.emplace_back(u, v, 0.f);
}

void load_normal(const std::string &line, std::vector<Vec3> &normals) {
  float x, y, z;
  sscanf(line.c_str(), "vn %f %f %f", &x, &y, &z);
  normals.emplace_back(x, y, z);
}

void clean(int e[9], int n) {
  for (int i = 0; i < 9; i++) {
    e[i] = (e[i] < 0 ? n + e[i] : e[i] - 1);
  }
}

void clean_alt(int e[9], int n1, int n2) {
  for (int i = 0; i < 9; i++) {
    e[i] = (e[i] < 0 ? ((i % 2 == 0) ? n1 : n2) + e[i] : e[i] - 1);
  }
}

void load_face(const std::string &line, std::string &current_name, const std::vector<Vec3> &verts, const std::vector<Vec3> &normals, const std::map<std::string, BSDF> materials, BSDF *mat_arr, std::vector<Primitive> &prims) {
  int mat_idx = std::distance(materials.begin(), materials.find(current_name));
  if (mat_idx == materials.size()) {
    printf("No material with name %s\n", current_name.c_str());
    current_name = "";
    return load_face(line, current_name, verts, normals, materials, mat_arr, prims);
  }
  BSDF *bsdf = mat_arr + mat_idx;

  int n;
  int e[9];

  n = sscanf(line.c_str(), "f %d %d %d %d", e+0, e+1, e+2, e+3);
  if (n >= 3) {
    clean(e, verts.size());
    prims.emplace_back(Tri(verts[e[0]], verts[e[1]], verts[e[2]]), bsdf);
    if (n == 4) prims.emplace_back(Tri(verts[e[1]], verts[e[2]], verts[e[3]]), bsdf);
    return;
  }

  n = sscanf(line.c_str(), "f %d/%*d %d/%*d %d/%*d %d/%*d", e+0, e+1, e+2, e+3);
  if (n >= 3) {
    clean(e, verts.size());
    prims.emplace_back(Tri(verts[e[0]], verts[e[1]], verts[e[2]]), bsdf);
    if (n == 4) prims.emplace_back(Tri(verts[e[1]], verts[e[2]], verts[e[3]]), bsdf);
    return;
  }

  n = sscanf(line.c_str(), "f %d/%*d/%d %d/%*d/%d %d/%*d/%d %d/%*d/%d", e+0, e+1, e+2, e+3, e+4, e+5, e+6, e+7);
  if (n >= 6) {
    clean_alt(e, verts.size(), normals.size());
    prims.emplace_back(Tri(verts[e[0]], verts[e[2]], verts[e[4]], normals[e[1]], normals[e[3]], normals[e[5]]), bsdf);
    if (n == 8) prims.emplace_back(Tri(verts[e[2]], verts[e[4]], verts[e[6]], normals[e[3]], normals[e[5]], normals[e[7]]), bsdf);
    return;
  }

  n = sscanf(line.c_str(), "f %d//%d %d//%d %d//%d %d//%d", e+0, e+1, e+2, e+3, e+4, e+5, e+6, e+7);
  if (n >= 6) {
    clean_alt(e, verts.size(), normals.size());
    prims.emplace_back(Tri(verts[e[0]], verts[e[2]], verts[e[4]], normals[e[1]], normals[e[3]], normals[e[5]]), bsdf);
    if (n > 6) prims.emplace_back(Tri(verts[e[2]], verts[e[4]], verts[e[6]], normals[e[3]], normals[e[5]], normals[e[7]]), bsdf);
    return;
  }

  printf("Cannot parse line: %s\n", line.c_str());
  exit(1);
}

void load_obj(std::string fname, Scene *scene) {
  std::map<std::string, BSDF> materials;
  materials[""] = BSDF(new Lambertian(Vec3(1.f, 1.f, 1.f)));
  std::vector<Texture> textures;

  std::ifstream input(fname);
  fname.replace(fname.end() - 3, fname.end(), "mtl");
  load_materials(fname, materials, textures);

  int n_mats = materials.size();
  BSDF *mats = (BSDF *)malloc(n_mats * sizeof(BSDF));
  int i = 0;
  for (auto it = materials.begin(); it != materials.end(); it++, i++) mats[i] = it->second; 

  int n_textures = textures.size();
  Texture *tex_arr = (Texture *)malloc(n_textures * sizeof(Texture));
  for (i = 0; i < textures.size(); i++) tex_arr[i] = textures[i];

  std::string current_name = "";

  std::vector<Vec3> verts;
  std::vector<Vec3> texture_coords;
  std::vector<Vec3> normals;
  std::vector<Primitive> prims;

  for (std::string line; std::getline(input, line); ) {
    line = trim(line);
    if (line[0] == '#' || line == "" || line == "\n" || line[0] == 'g' || line[0] == 'o' || line[0] == 's') {
      continue;
    } else if (line[0] == 'v' && line[1] == ' ') {
      load_vertex(line, verts);
    } else if (line[0] == 'v' && line[1] == 't') {
      load_vertex_tex(line, texture_coords);
    } else if (line[0] == 'v' && line[1] == 'n') {
      load_normal(line, normals);
    } else if (line.find("usemtl") != std::string::npos) {
      current_name = line.substr(7);
    } else if (line[0] == 'f') {
      load_face(line, current_name, verts, normals, materials, mats, prims);
    } else {
      printf("Unrecognized line %s\n", line.c_str());
    }
  }

  Primitive *prim_arr = (Primitive *)malloc(prims.size() * sizeof(Primitive));

  int n_lights = 0;
  for (int i = 0; i < prims.size(); i++) {
    prim_arr[i] = prims[i];
    if (prims[i].bsdf->is_light()) n_lights++;
  }

  BVH bvh = build_bvh(prim_arr, prims.size());

  int light = 0;
  Primitive **lights = (Primitive **)malloc(n_lights * sizeof(Primitive *));
  for (int i = 0; i < prims.size(); i++) {
    if (prim_arr[i].bsdf->is_light()) lights[light++] = prim_arr + i;
  }

  scene->b = bvh;
  scene->lights = lights;
  scene->n_lights = n_lights;
  scene->materials = mats;
  scene->n_materials = n_mats;
  scene->textures = tex_arr;
  scene->n_textures = n_textures;
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
