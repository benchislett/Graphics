cmake_minimum_required(VERSION 3.0)

project(benpt LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "60")

include_directories("src/")

set(CMAKE_CXX_FLAGS "-Wall -std=c++20")
set(CMAKE_CUDA_FLAGS "--use_fast_math")

file(GLOB_RECURSE LIB_SOURCES src/*.cu)
file(GLOB_RECURSE TEST_SOURCES test/*.cu)

add_library(benpt STATIC ${LIB_SOURCES})
target_link_libraries(benpt PUBLIC CUDA::cudart)

set_target_properties(benpt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

find_package(GTest)
if (${GTEST_FOUND})
  add_executable(test ${TEST_SOURCES})
  target_link_libraries(test PRIVATE benpt)
  target_link_libraries(test PRIVATE GTest::GTest GTest::Main)
endif()

