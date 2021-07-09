#include "triangle.cuh"

#include <catch2/catch.hpp>
#include <numbers>

using std::numbers::inv_sqrt3;
using std::numbers::sqrt2;
constexpr double inv_sqrt2 = 1.0 / sqrt2;
constexpr double inv_sqrt5 = 1.0 / sqrt(5.0);

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

TEST_CASE("Uniform normal", "[TriangleNormals.at(float3)]") {
  TriangleNormals n({1, 0, 0});

  CHECK(n.at({1, 0, 0}).x == Approx(1));
  CHECK(n.at({1, 0, 0}).y == Approx(0));
  CHECK(n.at({1, 0, 0}).z == Approx(0));

  CHECK(n.at({0.5, 0.5, 0}).x == Approx(1));
  CHECK(n.at({0.5, 0.5, 0}).y == Approx(0));
  CHECK(n.at({0.5, 0.5, 0}).z == Approx(0));

  CHECK(n.at({0.5, 0.3, 0.2}).x == Approx(1));
  CHECK(n.at({0.5, 0.3, 0.2}).y == Approx(0));
  CHECK(n.at({0.5, 0.3, 0.2}).z == Approx(0));
}

TEST_CASE("Aligned normal", "[TriangleNormals.at(float3)]") {
  TriangleNormals n({1, 0, 0}, {1, 0, 1}, {0, 0, 1});

  CHECK(n.at({1, 0, 0}).x == Approx(inv_sqrt2));
  CHECK(n.at({1, 0, 0}).y == Approx(0));
  CHECK(n.at({1, 0, 0}).z == Approx(inv_sqrt2));

  CHECK(n.at({0.5, 0.5, 0}).x == Approx(inv_sqrt5));
  CHECK(n.at({0.5, 0.5, 0}).y == Approx(0));
  CHECK(n.at({0.5, 0.5, 0}).z == Approx(2 * inv_sqrt5));

  CHECK(n.at({0.5, 0.3, 0.2}).x == Approx(7 / sqrt(113)));
  CHECK(n.at({0.5, 0.3, 0.2}).y == Approx(0));
  CHECK(n.at({0.5, 0.3, 0.2}).z == Approx(8 / sqrt(113)));
}

TEST_CASE("Unaligned normal", "[TriangleNormals.at(float3, Ray)]") {
  TriangleNormals n({1, 0, 0}, {1, 0, 1}, {0, 0, 1});
  Ray r({0, 0, 0}, {1, -1, 1});

  CHECK(n.at({1, 0, 0}, r).x == Approx(-inv_sqrt2));
  CHECK(n.at({1, 0, 0}, r).y == Approx(0));
  CHECK(n.at({1, 0, 0}, r).z == Approx(-inv_sqrt2));

  CHECK(n.at({0.5, 0.5, 0}, r).x == Approx(-inv_sqrt5));
  CHECK(n.at({0.5, 0.5, 0}, r).y == Approx(0));
  CHECK(n.at({0.5, 0.5, 0}, r).z == Approx(-2 * inv_sqrt5));

  CHECK(n.at({0.5, 0.3, 0.2}, r).x == Approx(-7 / sqrt(113)));
  CHECK(n.at({0.5, 0.3, 0.2}, r).y == Approx(0));
  CHECK(n.at({0.5, 0.3, 0.2}, r).z == Approx(-8 / sqrt(113)));
}
