#include "camera.h"

#include "gtest/gtest.h"

TEST(Camera_Sample, Init_And_Corners) {
  float fov    = M_PI / 2.f; // 90 degrees
  float aspect = 4.f;

  float3 position = make_float3(100.f, 70.f, 33.f);
  float3 target   = make_float3(200.f, 70.f, 33.f);

  Camera camera = make_camera(position, target, fov, aspect);

  EXPECT_NEAR(camera.position.x, position.x, 0.00001f);
  EXPECT_NEAR(camera.position.y, position.y, 0.00001f);
  EXPECT_NEAR(camera.position.z, position.z, 0.00001f);

  EXPECT_NEAR(camera.horizontal.x, 0.f, 0.00001f);
  EXPECT_NEAR(camera.horizontal.y, 0.f, 0.00001f);
  EXPECT_NEAR(camera.horizontal.z, 8.f, 0.00001f);

  EXPECT_NEAR(camera.vertical.x, 0.f, 0.00001f);
  EXPECT_NEAR(camera.vertical.y, 2.f, 0.00001f);
  EXPECT_NEAR(camera.vertical.z, 0.f, 0.00001f);

  Ray c0 = get_ray(camera, 0.f, 0.f);
  Ray c1 = get_ray(camera, 0.f, 1.f);
  Ray c2 = get_ray(camera, 1.f, 0.f);
  Ray c3 = get_ray(camera, 1.f, 1.f);

  EXPECT_NEAR(c0.origin.x, position.x, 0.00001f);
  EXPECT_NEAR(c0.origin.y, position.y, 0.00001f);
  EXPECT_NEAR(c0.origin.z, position.z, 0.00001f);

  EXPECT_NEAR(c1.origin.x, position.x, 0.00001f);
  EXPECT_NEAR(c1.origin.y, position.y, 0.00001f);
  EXPECT_NEAR(c1.origin.z, position.z, 0.00001f);

  EXPECT_NEAR(c2.origin.x, position.x, 0.00001f);
  EXPECT_NEAR(c2.origin.y, position.y, 0.00001f);
  EXPECT_NEAR(c2.origin.z, position.z, 0.00001f);

  EXPECT_NEAR(c3.origin.x, position.x, 0.00001f);
  EXPECT_NEAR(c3.origin.y, position.y, 0.00001f);
  EXPECT_NEAR(c3.origin.z, position.z, 0.00001f);
}
