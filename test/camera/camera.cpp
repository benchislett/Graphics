#include "camera.cuh"

#include <catch2/catch.hpp>
#include <numbers>

using std::numbers::pi;

TEST_CASE("Positive ray from camera", "[Camera.get_ray]") {
  Camera cam(pi / 2, 1.0, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});

  CHECK(cam.get_ray(0.0, 0.0).d.x == Approx(0.97728f).margin(0.0001));
  CHECK(cam.get_ray(0.0, 0.0).d.y == Approx(-0.13807f).margin(0.0001));
  CHECK(cam.get_ray(0.0, 0.0).d.z == Approx(0.16079f).margin(0.0001));

  CHECK(cam.get_ray(1.0, 0.0).d.x == Approx(0.16079f).margin(0.0001));
  CHECK(cam.get_ray(1.0, 0.0).d.y == Approx(-0.13807f).margin(0.0001));
  CHECK(cam.get_ray(1.0, 0.0).d.z == Approx(0.97728f).margin(0.0001));

  CHECK(cam.get_ray(0.0, 1.0).d.x == Approx(0.50588f).margin(0.0001));
  CHECK(cam.get_ray(0.0, 1.0).d.y == Approx(0.80474f).margin(0.0001));
  CHECK(cam.get_ray(0.0, 1.0).d.z == Approx(-0.31062f).margin(0.0001));

  CHECK(cam.get_ray(1.0, 1.0).d.x == Approx(-0.31062f).margin(0.0001));
  CHECK(cam.get_ray(1.0, 1.0).d.y == Approx(0.80474f).margin(0.0001));
  CHECK(cam.get_ray(1.0, 1.0).d.z == Approx(0.50588f).margin(0.0001));
}

TEST_CASE("Negative ray from camera", "[Camera.get_ray]") {
  Camera cam(pi / 2, 1.0, {0.0, 0.0, 0.0}, {-1.0, -1.0, -1.0});

  CHECK(cam.get_ray(0.0, 1.0).d.x == Approx(-0.97728f).margin(0.0001));
  CHECK(cam.get_ray(0.0, 1.0).d.y == Approx(0.13807f).margin(0.0001));
  CHECK(cam.get_ray(0.0, 1.0).d.z == Approx(-0.16079f).margin(0.0001));

  CHECK(cam.get_ray(1.0, 1.0).d.x == Approx(-0.16079f).margin(0.0001));
  CHECK(cam.get_ray(1.0, 1.0).d.y == Approx(0.13807f).margin(0.0001));
  CHECK(cam.get_ray(1.0, 1.0).d.z == Approx(-0.97728f).margin(0.0001));

  CHECK(cam.get_ray(0.0, 0.0).d.x == Approx(-0.50588f).margin(0.0001));
  CHECK(cam.get_ray(0.0, 0.0).d.y == Approx(-0.80474f).margin(0.0001));
  CHECK(cam.get_ray(0.0, 0.0).d.z == Approx(0.31062f).margin(0.0001));

  CHECK(cam.get_ray(1.0, 0.0).d.x == Approx(0.31062f).margin(0.0001));
  CHECK(cam.get_ray(1.0, 0.0).d.y == Approx(-0.80474f).margin(0.0001));
  CHECK(cam.get_ray(1.0, 0.0).d.z == Approx(-0.50588f).margin(0.0001));
}
