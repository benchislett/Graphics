#include "camera.h"
#include "cu_math.h"
#include "geometry.h"
#include "integrate.h"
#include "scene.h"

#include <SFML/Graphics.hpp>
#include <cassert>
#include <iostream>
#include <string>

constexpr int width    = 256;
constexpr int height   = 256;
constexpr float aspect = (float) width / (float) height;

Scene scene;
float3 position       = make_float3(0.f);
float3 look_direction = make_float3(1.f, 0.f, 0.f);

void rotate_xz(float theta) {
  float x          = look_direction.x;
  float z          = look_direction.z;
  look_direction.x = x * cosf(theta) - z * sinf(theta);
  look_direction.z = z * cosf(theta) - x * sinf(theta);
}

int main(int argc, char** argv) {
  scene = from_obj("data/test.obj");

  sf::RenderWindow window(sf::VideoMode(width, height), "Demo Window");

  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::KeyPressed) {
        if (event.key.code == sf::Keyboard::Left) {
          rotate_xz(0.523599f);
        } else if (event.key.code == sf::Keyboard::Right) {
          rotate_xz(-0.523599f);
        } else if (event.key.code == sf::Keyboard::W) {
          position.z += 0.5f;
        } else if (event.key.code == sf::Keyboard::S) {
          position.z -= 0.5f;
        } else if (event.key.code == sf::Keyboard::A) {
          position.x += 0.5f;
        } else if (event.key.code == sf::Keyboard::D) {
          position.x -= 0.5f;
        } else if (event.key.code == sf::Keyboard::Space) {
          position.y += 0.5f;
        } else if (event.key.code == sf::Keyboard::LShift) {
          position.y -= 0.5f;
        }
      }
      if (event.type == sf::Event::Closed) {
        window.close();
      }
    }

    Camera camera = make_camera(position, look_direction - position, 1.57f, aspect);
    Image image   = render(camera, scene, width, height, 1);

    sf::Image image_data;
    image_data.create(width, height, (const sf::Uint8*) image.data);

    sf::Texture image_tex;
    image_tex.loadFromImage(image_data);

    sf::Sprite image_sprite;
    image_sprite.setTexture(image_tex, true);
    std::cout << image_sprite.getPosition().x << '\n';
    std::cout << image_sprite.getPosition().y << "\n\n";

    window.clear();
    window.draw(image_sprite);
    window.display();
  }

  return 0;
}

