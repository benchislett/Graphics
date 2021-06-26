cmake_minimum_required(VERSION 3.0)

project(benpt LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)
find_package(Catch2)
find_package(embree 3.0)

set(CMAKE_CUDA_ARCHITECTURES "60")

include_directories("include/geometry")
include_directories("include/math")
include_directories("src/")

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CUDA_FLAGS "")

file(GLOB_RECURSE LIB_SOURCES src/*.cu)
file(GLOB_RECURSE TEST_SOURCES test/*.cpp)

add_library(benpt STATIC ${LIB_SOURCES})
target_link_libraries(benpt PUBLIC CUDA::cudart)

set_target_properties(benpt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(benpt_test ${TEST_SOURCES})
target_link_libraries(benpt_test PRIVATE benpt)
target_link_libraries(benpt_test PRIVATE embree)
target_link_libraries(benpt_test PRIVATE Catch2::Catch2)

include(CTest)
include(Catch)
catch_discover_tests(benpt_test)

