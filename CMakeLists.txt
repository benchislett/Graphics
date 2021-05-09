cmake_minimum_required(VERSION 3.0)

project(benpt LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "60")

include_directories("src/")
include_directories("src/scene/")
include_directories("src/image/")
include_directories("src/camera/")
include_directories("src/geometry/")
include_directories("src/material/")
include_directories("src/cuda_misc/")
include_directories("src/integrator/")

set(CMAKE_CXX_FLAGS "-Wall -std=c++20")
# set(CMAKE_CUDA_FLAGS "-Wall -ffast-math")

set(DEMO_SOURCES demo/main.cpp)
set(TEST_SOURCES test/geometry/ray_triangle.cpp test/camera/sample.cpp)

set(LIB_GEOMETRY_SOURCES src/geometry/ray_aabb.cu src/geometry/ray_triangle.cu src/geometry/traverse.cu src/geometry/triangle.cu)
set(LIB_CAMERA_SOURCES src/camera/init.cu src/camera/sample.cu)
set(LIB_SCENE_SOURCES src/scene/scene.cu src/scene/from_obj.cu)
set(LIB_IMAGE_SOURCES src/image/to_ppm.cu)
set(LIB_INTEGRATOR_SOURCES src/integrator/path_trace.cu)
set(LIB_SOURCES ${LIB_GEOMETRY_SOURCES} ${LIB_CAMERA_SOURCES} ${LIB_SCENE_SOURCES} ${LIB_IMAGE_SOURCES} ${LIB_INTEGRATOR_SOURCES})

add_library(benpt STATIC ${LIB_SOURCES})
target_link_libraries(benpt PUBLIC CUDA::cudart)

set_target_properties(benpt
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

find_package(SFML 2.5 COMPONENTS graphics)
if (${SFML_FOUND})
  add_executable(demo ${DEMO_SOURCES})
  target_link_libraries(demo PRIVATE benpt)
  target_link_libraries(demo PRIVATE sfml-graphics)
endif()

find_package(GTest)
if (${GTEST_FOUND})
  add_executable(test ${TEST_SOURCES})
  target_link_libraries(test PRIVATE benpt)
  target_link_libraries(test PRIVATE GTest::GTest GTest::Main)
endif()

