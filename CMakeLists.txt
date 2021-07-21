cmake_minimum_required(VERSION 3.0)

project(benpt LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)
find_package(Catch2)
find_package(lodepng CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "60")

include_directories("include/geometry")
include_directories("include/camera")
include_directories("include/render")
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

target_compile_options(benpt BEFORE PRIVATE -Wall -Wextra)

set_target_properties(benpt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

### SCRIPTS ###

add_executable(render_sphere scripts/render_sphere.cpp)
target_link_libraries(render_sphere PRIVATE benpt)

### PYBIND

if(SKBUILD)
  # Scikit-Build does not add your site-packages to the search path automatically,
  # so we need to add it _or_ the pybind11 specific directory here.
  execute_process(
    COMMAND
      "${PYTHON_EXECUTABLE}" -c
      "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
endif()

pybind11_add_module(benptpy_core MODULE src/pybind/bindings.cpp)
target_link_libraries(benptpy_core PRIVATE benpt)

install(TARGETS benptpy_core DESTINATION .)

### TESTS ###

add_executable(benpt_test ${TEST_SOURCES})
target_link_libraries(benpt_test PRIVATE benpt)
target_link_libraries(benpt_test PRIVATE Catch2::Catch2)

include(CTest)
include(Catch)
catch_discover_tests(benpt_test)
