#include "image.cuh"

#include <catch2/catch.hpp>

TEST_CASE("Reads from PNG", "[Image]") {
  Image img("../test/image/test_data/test.png");
  REQUIRE(img.width == 3);
  REQUIRE(img.height == 4);
  CHECK(img[0].x == 1.0);
  CHECK(img[0].y == 0.0);
  CHECK(img[0].z == 0.0);

  CHECK(img[3].x == Approx(0.5).margin(1.0 / 255.));
  CHECK(img[3].y == 0.0);
  CHECK(img[3].z == 0.0);
}

TEST_CASE("Writes to PNG", "[Image]") {
  Image img(3, 4);
  img[0] = {1.0, 0.0, 0.0};
  img[1] = {0.0, 1.0, 0.0};
  img[2] = {0.0, 0.0, 1.0};

  img[3] = {0.5, 0.0, 0.0};
  img[4] = {0.0, 0.5, 0.0};
  img[5] = {0.0, 0.0, 0.5};

  img[6] = {0.0, 0.0, 0.0};
  img[7] = img[6];
  img[8] = img[7];

  img[9]  = {1.0, 1.0, 1.0};
  img[10] = img[9];
  img[11] = img[10];

  img.to_png("../test/image/test_data/test_out.png");

  Image other("../test/image/test_data/test_out.png");
  REQUIRE(other.width == 3);
  REQUIRE(other.height == 4);
}