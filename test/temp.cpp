#include "temp.cuh"

#include "gtest/gtest.h"

TEST(Geometry, Miss) {
  float v0 = 0, v1 = 0, v2 = 0;
  float o0 = 0, o1 = 0, o2 = 0;
  float d0 = 0, d1 = 0, d2 = 0;
  EXPECT_EQ(true, hit(v0, v1, v2, o0, o1, o2, d0, d1, d2));
}