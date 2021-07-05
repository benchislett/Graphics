#define CATCH_CONFIG_MAIN
#include "aabb.cuh"

#include <catch2/catch.hpp>

TEST_CASE("Easterly Ray hits Eastern AABB", "[AABB.intersects(Ray)]") {
  Ray r({0, 0, 0}, {1, 0, 0});
  AABB b({1, -1, -1}, {3, 1, 1}); // 2x2x2 about (2, 0, 0)

  REQUIRE(b.intersects(r).hit);
}

TEST_CASE("Easterly Ray misses Western AABB", "[AABB.intersects(Ray)]") {
  Ray r({0, 0, 0}, {1, 0, 0});
  AABB b({-3, -1, -1}, {-1, 1, 1}); // 2x2x2 about (-2, 0, 0)

  REQUIRE_FALSE(b.intersects(r).hit);
}

TEST_CASE("Westerly Ray hits Western AABB", "[AABB.intersects(Ray)]") {
  Ray r({0, 0, 0}, {-1, 0, 0});
  AABB b({-3, -1, -1}, {-1, 1, 1}); // 2x2x2 about (-2, 0, 0)

  REQUIRE(b.intersects(r).hit);
}

TEST_CASE("Westerly Ray misses Eastern AABB", "[AABB.intersects(Ray)]") {
  Ray r({0, 0, 0}, {-1, 0, 0});
  AABB b({1, -1, -1}, {3, 1, 1}); // 2x2x2 about (2, 0, 0)

  REQUIRE_FALSE(b.intersects(r).hit);
}
