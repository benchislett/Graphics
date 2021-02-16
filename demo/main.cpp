#include "camera.cuh"
#include "cu_math.cuh"
#include "geometry.cuh"
#include "integrate.cuh"
#include "scene.cuh"

#include <SFML/Graphics.hpp>
#include <cassert>
#include <iostream>
#include <string>

constexpr int width    = 1024;
constexpr int height   = 1024;
constexpr float aspect = (float) width / (float) height;

Scene scene;
float3 position       = make_float3(0.f);
float3 look_direction = make_float3(0.f, 0.f, -1.f);

Image image;
int main(int argc, char** argv) {
  scene = from_obj("data/cornell.obj");

  sf::RenderWindow window(sf::VideoMode(width, height), "Demo Window", sf::Style::Close);

  sf::Clock clock;
  sf::Font font;
  if (!font.loadFromFile("/usr/share/fonts/TTF/DejaVuSans.ttf")) {
    fprintf(stderr, "Error loading font!\n");
  }


  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::KeyPressed) {
        float pan_factor = 0.5f;
        if (event.key.control) {
          pan_factor /= 10.f;
        }
        if (event.key.code == sf::Keyboard::W) {
          position.z -= pan_factor;
        } else if (event.key.code == sf::Keyboard::S) {
          position.z += pan_factor;
        } else if (event.key.code == sf::Keyboard::A) {
          position.x -= pan_factor;
        } else if (event.key.code == sf::Keyboard::D) {
          position.x += pan_factor;
        } else if (event.key.code == sf::Keyboard::Space) {
          position.y += pan_factor;
        } else if (event.key.code == sf::Keyboard::LShift) {
          position.y -= pan_factor;
        }
      }
      if (event.type == sf::Event::Closed) {
        window.close();
        to_ppm(image, "output/cornell.ppm");
      }
    }

    Camera camera = make_camera(position, look_direction + position, 1.57f, aspect);
    image         = render(camera, scene, width, height, 1);

    sf::Image image_data;
    image_data.create(width, height, (const sf::Uint8*) image.data);

    sf::Texture image_tex;
    image_tex.loadFromImage(image_data);

    sf::Sprite image_sprite;
    image_sprite.setTexture(image_tex, true);

    float fps = 1.f / clock.getElapsedTime().asSeconds();
    clock.restart();
    sf::Text fps_text;
    fps_text.setFont(font);
    fps_text.setString("FPS: " + std::to_string(fps));
    fps_text.setCharacterSize(24);
    fps_text.setFillColor(sf::Color::Red);

    window.clear();
    window.draw(image_sprite);
    window.draw(fps_text);
    window.display();

    free(image.data);
  }

  return 0;
}
