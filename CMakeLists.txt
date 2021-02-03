cmake_minimum_required(VERSION 3.0)

project(benpt LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)

include_directories("${CUDA_INCLUDE_DIRS}")
include_directories("src/")
include_directories("src/scene/")
include_directories("src/camera/")
include_directories("src/geometry/")
include_directories("src/integrator/")

set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
set(CMAKE_CXX_FLAGS    "-Wall -std=c++20 -Ofast -flto")

set(DEMO_SOURCES demo/main.cpp)
set(TEST_SOURCES test/geometry/ray_triangle.cpp test/camera/sample.cpp)

set(LIB_GEOMETRY_SOURCES src/geometry/ray_triangle.cpp src/geometry/ray_aabb.cpp src/geometry/traverse.cpp)
set(LIB_CAMERA_SOURCES src/camera/init.cpp src/camera/sample.cpp)
set(LIB_SCENE_SOURCES src/scene/from_obj.cpp src/scene/to_ppm.cpp)
set(LIB_INTEGRATOR_SOURCES src/integrator/path_trace.cpp)
set(LIB_SOURCES ${LIB_GEOMETRY_SOURCES} ${LIB_CAMERA_SOURCES} ${LIB_SCENE_SOURCES} ${LIB_INTEGRATOR_SOURCES})

add_library(benpt STATIC ${LIB_SOURCES})

set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL)
find_package(GLUT)
find_package(GLEW)
if (${OpenGL_FOUND} AND ${GLEW_FOUND})
  add_executable(demo ${DEMO_SOURCES})
  target_link_libraries(demo PRIVATE benpt)
  include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
  target_link_libraries(demo PRIVATE GLEW::GLEW ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
endif()

find_package(GTest)
if (${GTEST_FOUND})
  add_executable(test ${TEST_SOURCES})
  target_link_libraries(test PRIVATE benpt)
  target_link_libraries(test PRIVATE GTest::GTest GTest::Main)
endif()

