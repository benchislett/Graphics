cmake_minimum_required(VERSION 3.0)

project(benpt LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)
find_package(Catch2)
find_package(lodepng CONFIG REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "60")

include_directories("include/geometry")
include_directories("include/camera")
include_directories("include/memory")
include_directories("include/render")
include_directories("include/timing")
include_directories("include/image")
include_directories("include/math")
include_directories("include/")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 20)

### CORE LIBRARY ###

file(GLOB_RECURSE LIB_SOURCES src/*.cu)
file(GLOB_RECURSE TEST_SOURCES test/*.cpp)

add_library(benpt STATIC ${LIB_SOURCES})
target_link_libraries(benpt PUBLIC CUDA::cudart)
target_link_libraries(benpt PUBLIC lodepng)

set_target_properties(benpt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

### SCRIPTS ###

add_executable(render_frontend scripts/render_frontend.cpp)
target_link_libraries(render_frontend PRIVATE benpt)

### TESTS ###

add_executable(benpt_test ${TEST_SOURCES})
target_link_libraries(benpt_test PRIVATE benpt)
target_link_libraries(benpt_test PRIVATE Catch2::Catch2)

include(CTest)
include(Catch)
catch_discover_tests(benpt_test)
