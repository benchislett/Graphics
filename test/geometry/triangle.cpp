#include "triangle.cuh"

#include <catch2/catch.hpp>

TEST_CASE("Easterly Ray hits Eastern Triangle", "[Triangle.intersects(Ray)]") {
  Ray r({0, 0, 0}, {1, 0, 0});
  Triangle t({2, -1, -1}, {2, 1, -1}, {2, 0, 1});

  REQUIRE(t.intersects(r).hit);
}

TEST_CASE("Easterly Ray misses Western Triangle", "[Triangle.intersects(Ray)]") {
  Ray r({0, 0, 0}, {1, 0, 0});
  Triangle t({-2, -1, -1}, {-2, 1, -1}, {-2, 0, 1});

  REQUIRE_FALSE(t.intersects(r).hit);
}

TEST_CASE("Westerly Ray hits Western Triangle", "[Triangle.intersects(Ray)]") {
  Ray r({0, 0, 0}, {-1, 0, 0});
  Triangle t({-2, -1, -1}, {-2, 1, -1}, {-2, 0, 1});

  REQUIRE(t.intersects(r).hit);
}

TEST_CASE("Westerly Ray misses Eastern Triangle", "[Triangle.intersects(Ray)]") {
  Ray r({0, 0, 0}, {-1, 0, 0});
  Triangle t({2, -1, -1}, {2, 1, -1}, {2, 0, 1});

  REQUIRE_FALSE(t.intersects(r).hit);
}
