cmake_minimum_required(VERSION 3.0)

project(benpt LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)

include_directories("${CUDA_INCLUDE_DIRS}")
include_directories("src/")
include_directories("src/geometry/")

set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
set(CMAKE_CXX_FLAGS    "-Wall -std=c++20 -Ofast -flto")

set(MAIN_SOURCE src/main.cpp)
set(LIB_SOURCES src/geometry/ray_triangle.cpp src/geometry/ray_aabb.cpp)

add_library(benpt STATIC ${LIB_SOURCES})

add_executable(main ${MAIN_SOURCE})
target_link_libraries(main PUBLIC benpt)

