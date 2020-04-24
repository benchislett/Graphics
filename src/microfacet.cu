#include "microfacet.cuh"
#include "onb_math.cuh"

float MicrofacetDistribution::g1(const Vec3 &w) const {
  return 1.f / (1.f + lambda(w));
}

float MicrofacetDistribution::g(const Vec3 &wo, const Vec3 &wi) const {
  return 1.f / (1.f + lambda(wo) + lambda(wi));
}

float MicrofacetDistribution::pdf(const Vec3 &wo, const Vec3 &wh) const {
  if (sample_visible_area) {
    return d(wh) * g1(wo) * dot_abs(wo, wh) / abscos_theta(wo);
  } else {
    return d(wh) * abscos_theta(wh);
  }
}

float alpha(float roughness) {
  float x = logf(fmax(0.001f, roughness));
  return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
}

Beckmann::Beckmann(float roughness, bool s) : MicrofacetDistribution(s) {
  alpha_x = alpha(roughness);
  alpha_y = alpha_x;
}

void beckmann_sample11(float costhetai, float u, float v, float *slope_x, float *slope_y) {
  if (costhetai > 0.9999) {
    float r = sqrtf(-logf(1.f - u));
    float sinphi = sinf(TWO_PI * v);
    float cosphi = cosf(TWO_PI * v);
    *slope_x = r * cosphi;
    *slope_y = r * sinphi;
    return;
  }

  float sinthetai = sqrtf(fmax(0.f, 1.f - costhetai * costhetai));
  float tanthetai = sinthetai / costhetai;
  float cotthetai = 1.f / tanthetai;

  float a = -1;
  float c = erff(cotthetai);
  float sample_x = fmax(u, 0.000001f);
  
  float thetai = acosf(costhetai);
  float fit = 1.f + thetai * (-0.867f + thetai * (0.4265f - 0.0594f * thetai));
  float b = c - (1.f + c) * powf(1.f - sample_x, fit);

  float norm = 1.f / (1.f + c + SQRT_INV_PI * tanthetai * expf(-cotthetai * cotthetai));

  for (int i = 1; i < 10; i++) {
    if (!(b >= a && b <= c)) b = 0.5f * (a + c);

    float ei = erfinvf(b);
    float v = norm * (1.f + b + SQRT_INV_PI * tanthetai * expf(-ei * ei)) - sample_x;
    float d = norm * (1.f - ei * tanthetai);

    if (fabs(v) < 0.00001f) break;
    if (v > 0.f) c = b;
    else a = b;

    b -= v / d;
  }

  *slope_x = erfinvf(b);
  *slope_y = erfinvf(2.f * fmax(v, 0.000001f) - 1.f);
}

Vec3 beckmann_sample(const Vec3 &wi, float alpha_x, float alpha_y, float u, float v) {
  Vec3 wi_s = normalized(Vec3(alpha_x * wi.e[0], alpha_y * wi.e[1], wi.e[2]));

  float slope_x, slope_y;
  beckmann_sample11(cos_theta(wi_s), u, v, &slope_x, &slope_y);

  float tmp = cos_phi(wi_s) * slope_x - sin_phi(wi_s) * slope_y;
  slope_y = sin_phi(wi_s) * slope_x + cos_phi(wi_s) * slope_y;
  slope_x = tmp;

  slope_x *= alpha_x;
  slope_y *= alpha_y;

  return normalized(Vec3(-slope_x, -slope_y, 1.f));
}

Vec3 Beckmann::sample_wh(const Vec3 &wo, float u, float v) const {
  if (!sample_visible_area) {
    float tan2theta, phi;
    if (alpha_x == alpha_y) {
      float logsample = logf(1.f - u);
      tan2theta = -alpha_x * alpha_x * logsample;
      phi = v * TWO_PI;
    } else {
      float logsample = logf(1.f - u);
      phi = atanf(alpha_y / alpha_x * tanf(TWO_PI * v + PI_OVER_2));
    }
    float costheta = 1.f / sqrtf(1.f + tan2theta);
    float sintheta = sqrtf(fmax(0.f, 1.f - costheta * costheta));
    Vec3 wh = Vec3(sintheta * cosf(phi), sintheta * sinf(phi), costheta);
    if (!same_hemisphere(wo, wh)) wh *= -1.f;
    return wh;
  } else {
    bool flip = wo.e[2] < 0.f;
    Vec3 wh = beckmann_sample(flip ? (-1 * wo) : wo, alpha_x, alpha_y, u, v);
    if (flip) wh *= -1;
    return wh;
  }
}

float Beckmann::d(const Vec3 &wh) const {
  float tan2theta = tan2_theta(wh);
  if (isinf(tan2theta)) return 0.f;
  float cos4theta = cos2_theta(wh) * cos2_theta(wh);
  return expf(-tan2theta * (cos2_phi(wh) / (alpha_x * alpha_x) + sin2_phi(wh) / (alpha_y * alpha_y))) / (PI * alpha_x * alpha_x * cos4theta);
}

float Beckmann::lambda(const Vec3 &w) const {
  float abstantheta = fabs(tan_theta(w));
  if (isinf(abstantheta)) return 0.f;
  float alpha = sqrtf(cos2_phi(w) * alpha_x * alpha_x + sin2_phi(w) * alpha_y * alpha_y);
  float a = 1.f / (alpha * abstantheta);
  if (a >= 1.6f) return 0.f;
  return (1.f - 1.259f * a + 0.396f * a * a) / (3.535f * a + 2.181f * a * a);
}
