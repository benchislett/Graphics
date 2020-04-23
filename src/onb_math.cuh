#pragma once

#include "math.cuh"

float cos_theta(const Vec3 &w);
float cos2_theta(const Vec3 &w);
float abscos_theta(const Vec3 &w);
float sin_theta(const Vec3 &w);
float sin2_theta(const Vec3 &w);
float tan_theta(const Vec3 &w);
float tan2_theta(const Vec3 &w);

float cos_phi(const Vec3 &w);
float cos2_phi(const Vec3 &w);
float sin_phi(const Vec3 &w);
float sin2_phi(const Vec3 &w);

bool same_hemisphere(const Vec3 &w1, const Vec3 &w2);

Vec3 reflect(const Vec3 &w, const Vec3 &n);
bool refract(const Vec3 &w_in, const Vec3 &n, float eta, Vec3 *w_t);

Vec3 cosine_sample(float u, float v);

