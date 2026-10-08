cmake_minimum_required(VERSION 3.16)
project(demucs_mlx_worker C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wno-unused-parameter)

set(MLX_C_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

if(NOT MLX_C_ROOT)
  set(_mlx_c_guess
      "${CMAKE_CURRENT_SOURCE_DIR}/../../../../reading/music/mlx-c")
  if(EXISTS "${_mlx_c_guess}/CMakeLists.txt")
    set(MLX_C_ROOT "${_mlx_c_guess}")
  endif()
endif()

if(MLX_C_ROOT)
  message(STATUS "Using mlx-c at ${MLX_C_ROOT}")
  add_subdirectory("${MLX_C_ROOT}" "${CMAKE_BINARY_DIR}/mlx-c")
else()
  include(FetchContent)
  FetchContent_Declare(
    mlx-c
    GIT_REPOSITORY https://github.com/ml-explore/mlx-c.git
    GIT_TAG v0.7.0)
  FetchContent_MakeAvailable(mlx-c)
endif()

set(WORKER_SRC
    src/util.c
    src/audio.c
    src/config.c
    src/weights.c
    src/metal_kernels.c
    src/stft.c
    src/ops.c
    src/layers.c
    src/transformer.c
    src/htdemucs.c
    src/apply.c)

find_path(LAME_INCLUDE_DIR lame/lame.h
  PATHS /opt/homebrew/opt/lame/include /opt/homebrew/include /usr/local/include)
set(_dmx_lib_suffixes "${CMAKE_FIND_LIBRARY_SUFFIXES}")
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
find_library(LAME_LIBRARY NAMES mp3lame
  PATHS /opt/homebrew/opt/lame/lib /opt/homebrew/lib /usr/local/lib)
set(CMAKE_FIND_LIBRARY_SUFFIXES "${_dmx_lib_suffixes}")
unset(_dmx_lib_suffixes)
if(NOT LAME_INCLUDE_DIR OR NOT LAME_LIBRARY)
  message(FATAL_ERROR "libmp3lame not found (brew install lame)")
endif()
message(STATUS "libmp3lame: ${LAME_LIBRARY}")

add_library(demucs_c_core STATIC ${WORKER_SRC})
target_include_directories(demucs_c_core PUBLIC src "${LAME_INCLUDE_DIR}")
target_link_libraries(demucs_c_core PUBLIC mlxc mlx "${LAME_LIBRARY}")
set_target_properties(demucs_c_core PROPERTIES LINKER_LANGUAGE CXX)

add_executable(demucs_mlx_worker src/main.c)
target_link_libraries(demucs_mlx_worker PRIVATE demucs_c_core)
set_target_properties(demucs_mlx_worker PROPERTIES LINKER_LANGUAGE CXX)

add_executable(test_kernels src/test_kernels.c)
target_link_libraries(test_kernels PRIVATE demucs_c_core)
set_target_properties(test_kernels PROPERTIES LINKER_LANGUAGE CXX)

add_executable(test_stft src/test_stft.c)
target_link_libraries(test_stft PRIVATE demucs_c_core)
set_target_properties(test_stft PROPERTIES LINKER_LANGUAGE CXX)

add_executable(test_compare src/test_compare.c)
target_link_libraries(test_compare PRIVATE demucs_c_core)
set_target_properties(test_compare PROPERTIES LINKER_LANGUAGE CXX)

add_executable(test_ola src/test_ola.c)
target_link_libraries(test_ola PRIVATE demucs_c_core)
set_target_properties(test_ola PROPERTIES LINKER_LANGUAGE CXX)
