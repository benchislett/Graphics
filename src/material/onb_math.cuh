#pragma once

#include "cu_math.cuh"

IHD float cos_theta(float3 w) {
  return w.z;
}

IHD float abscos_theta(float3 w) {
  return fabsf(cos_theta(w));
}

IHD float cos2_theta(float3 w) {
  return cos_theta(w) * cos_theta(w);
}

IHD float sin2_theta(float3 w) {
  return fmaxf(0.f, 1.f - cos2_theta(w));
}

IHD float sin_theta(float3 w) {
  return sqrtf(sin2_theta(w));
}

IHD float tan_theta(float3 w) {
  return sin2_theta(w) / cos2_theta(w);
}

IHD float cos_phi(float3 w) {
  float sintheta = sin_theta(w);
  float x        = 0.f;
  if (sintheta != 0.f) {
    x = w.x / sintheta;
  }
  return clamp(x, -1.f, 1.f);
}

IHD float sin_phi(float3 w) {
  float sintheta = sin_theta(w);
  float x        = 0.f;
  if (sintheta != 0.f) {
    x = w.y / sintheta;
  }
  return clamp(x, -1.f, 1.f);
}

IHD float cos2_phi(float3 w) {
  float cosphi = cos_phi(w);
  return cosphi * cosphi;
}

IHD float sin2_phi(float3 w) {
  float sinphi = sin_phi(w);
  return sinphi * sinphi;
}

IHD float3 reflect(float3 w, float3 normal) {
  return 2 * dot(w, normal) * normal - w;
}

struct RefractionResult {
  float3 w_out;
  bool success;
};

IHD RefractionResult refract(float3 w_in, float3 normal, float eta) {
  float costheta_in   = dot(normal, w_in);
  float sin2theta_in  = fmaxf(0.f, 1.f - costheta_in * costheta_in);
  float sin2theta_out = eta * eta * sin2theta_in;

  if (sin2theta_out >= 1.f) {
    return {make_float3(0.f), false};
  }
  float costheta_out = sqrtf(1.f - sin2theta_out * sin2theta_out);

  return {(-eta * w_in) + (eta * costheta_in - costheta_out) * normal, true};
}

IHD bool same_hemisphere(float3 w1, float3 w2) {
  return cos_theta(w1) * cos_theta(w2) > 0.f;
}

IHD void concentric_sample_disk(float* u, float* v) {
  float uu = 2.f * (*u) - 1.f;
  float vv = 2.f * (*v) - 1.f;

  if (uu == 0.f || vv == 0.f) {
    *u = 0.f;
    *v = 0.f;
    return;
  }

  float r, theta;
  if (fabsf(uu) > fabsf(vv)) {
    r     = uu;
    theta = PI / 4.f * (vv / uu);
  } else {
    r     = vv;
    theta = PI / 2.f - PI / 4.f * (uu / vv);
  }

  *u = r * cosf(theta);
  *v = r * sinf(theta);
}

IHD float3 cosine_sample(float u, float v) {
  concentric_sample_disk(&u, &v);

  float z = sqrtf(fmaxf(0.f, 1.f - u * u - v * v));
  return make_float3(u, v, z);
}
