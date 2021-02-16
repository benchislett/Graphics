#include "geometry.cuh"

#include "gtest/gtest.h"

TEST(Ray_Triangle, Axis_Aligned) {
  Triangle tx = (Triangle){make_float3(4.f, -0.5f, -5.f), make_float3(4.f, -0.5f, 5.f), make_float3(4.f, 1.f, 0.f)};
  Triangle ty = (Triangle){make_float3(-0.5f, 4.f, -5.f), make_float3(-0.5f, 4.f, 5.f), make_float3(1.f, 4.f, 0.f)};
  Triangle tz = (Triangle){make_float3(-5.f, -0.5f, 4.f), make_float3(5.f, -0.5f, 4.f), make_float3(0.f, 1.f, 4.f)};

  Ray rx  = (Ray){make_float3(0.f), make_float3(1.f, 0.f, 0.f)};
  Ray ry  = (Ray){make_float3(0.f), make_float3(0.f, 1.f, 0.f)};
  Ray rz  = (Ray){make_float3(0.f), make_float3(0.f, 0.f, 1.f)};
  Ray rmx = (Ray){make_float3(0.f), make_float3(-1.f, 0.f, 0.f)};
  Ray rmy = (Ray){make_float3(0.f), make_float3(0.f, -1.f, 0.f)};
  Ray rmz = (Ray){make_float3(0.f), make_float3(0.f, 0.f, -1.f)};

  EXPECT_TRUE(hit(rx, tx).hit);
  EXPECT_TRUE(hit(ry, ty).hit);
  EXPECT_TRUE(hit(rz, tz).hit);
  EXPECT_NEAR(hit(rx, tx).time, 4.f, 0.000001f);
  EXPECT_NEAR(hit(ry, ty).time, 4.f, 0.000001f);
  EXPECT_NEAR(hit(rz, tz).time, 4.f, 0.000001f);

  EXPECT_FALSE(hit(rx, ty).hit);
  EXPECT_FALSE(hit(rx, tz).hit);

  EXPECT_FALSE(hit(ry, tx).hit);
  EXPECT_FALSE(hit(ry, tz).hit);

  EXPECT_FALSE(hit(rz, tx).hit);
  EXPECT_FALSE(hit(rz, ty).hit);

  EXPECT_FALSE(hit(rmx, tx).hit);
  EXPECT_FALSE(hit(rmy, ty).hit);
  EXPECT_FALSE(hit(rmz, tz).hit);
}

TEST(Ray_Triangle, Distant) {
  float3 v0 = make_float3(100.123f, -101.456f, 79.0f);
  float3 v1 = make_float3(99.1234f, -102.456f, 80.0f);
  float3 v2 = make_float3(99.7852f, -101.999f, 79.5f);

  Triangle tri = (Triangle){v0, v1, v2};

  float3 origin = make_float3(4.f, 2.f, 17.f);

  float3 hit_targets[3]  = {v0 + (0.1f * v1) + (0.1f * v2), v1 + (0.1f * v0) + (0.1f * v2),
                           v2 + (0.1f * v0) + (0.1f * v1)};
  float3 miss_targets[3] = {v0 - (0.1f * v1) - (0.1f * v2), v1 - (0.1f * v0) - (0.1f * v2),
                            v2 - (0.1f * v0) - (0.1f * v1)};

  auto test_hit = [=](const float3 target) {
    Ray ray = (Ray){origin, normalized(target - origin)};
    EXPECT_TRUE(hit(ray, tri).hit);
    EXPECT_NEAR(hit(ray, tri).time, length(target - origin), 0.1f);
  };

  auto test_miss = [=](const float3 target) {
    Ray ray = (Ray){origin, normalized(target - origin)};
    EXPECT_FALSE(hit(ray, tri).hit);
  };

  for (int i = 0; i < 3; i++) {
    test_hit(hit_targets[i] / 1.2f);
    test_miss(miss_targets[i] / 0.8f);
  }
}
