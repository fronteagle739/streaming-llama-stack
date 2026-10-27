cmake_minimum_required(VERSION 3.27)

project(omlx_glm_kernels LANGUAGES CXX)

# [2026-07-07] Default the library/runtime output dirs to a WRITABLE build path. When unset, the metallib
# custom-command wrote to ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/omlx_glm_kernels.metallib = "/omlx_glm_kernels.metallib"
# (filesystem root, read-only) -> air-lld "Read-only file system" build failure. Build proven on Mac Studio
# (Metal 32023.883, MLX libmlx.dylib, nanobind) with these defaults.
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

find_package(
  Python 3.8
  COMPONENTS Interpreter Development.Module
  REQUIRED)
# Match the installed MLX BINARY's nanobind ABI, not a stale source-tree pin.
# build.py inspects that ABI, selects isolated headers, then calls real ops.
if(NOT nanobind_ROOT)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    COMMAND_ERROR_IS_FATAL ANY
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT)
endif()
find_package(nanobind CONFIG REQUIRED)

if(NOT MLX_ROOT)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE MLX_ROOT)
endif()
find_package(MLX CONFIG REQUIRED)
list(GET MLX_INCLUDE_DIRS 0 OMLX_GLM_MLX_INCLUDE_DIR)

add_library(omlx_glm_kernel_ops)
set_target_properties(omlx_glm_kernel_ops PROPERTIES
  INSTALL_NAME_DIR "@rpath" BUILD_WITH_INSTALL_NAME_DIR ON)
target_sources(
  omlx_glm_kernel_ops
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/dsa_indexer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/exact_block_attention.cpp
  ${CMAKE_CURRENT_LIST_DIR}/fused_moe.cpp
  ${CMAKE_CURRENT_LIST_DIR}/sparse_mla.cpp)
target_include_directories(
  omlx_glm_kernel_ops
  PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(
  omlx_glm_kernel_ops
  PUBLIC OMLX_GLM_KERNEL_SOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}"
         OMLX_GLM_MLX_INCLUDE_DIR="${OMLX_GLM_MLX_INCLUDE_DIR}")
target_link_libraries(omlx_glm_kernel_ops PUBLIC mlx)

if(MLX_BUILD_METAL)
  set(OMLX_GLM_METAL_SOURCES
      ${CMAKE_CURRENT_LIST_DIR}/dsa_indexer.metal
      ${CMAKE_CURRENT_LIST_DIR}/exact_block_attention.metal
      ${CMAKE_CURRENT_LIST_DIR}/fused_moe.metal
      ${CMAKE_CURRENT_LIST_DIR}/sparse_mla.metal)
  file(GLOB_RECURSE OMLX_GLM_METAL_HEADERS CONFIGURE_DEPENDS
       ${CMAKE_CURRENT_LIST_DIR}/*.h)
  set(OMLX_GLM_METAL_FLAGS
      -x
      metal
      -Wall
      -Wextra
      -fno-fast-math
      -Wno-c++17-extensions
      -Wno-c++20-extensions)
  if(MLX_METAL_DEBUG)
    set(OMLX_GLM_METAL_FLAGS ${OMLX_GLM_METAL_FLAGS} -gline-tables-only
                             -frecord-sources)
  endif()
  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
    set(OMLX_GLM_METAL_FLAGS
        ${OMLX_GLM_METAL_FLAGS}
        "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
  endif()
  set(OMLX_GLM_METAL_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}
                                  ${MLX_INCLUDE_DIRS})
  set(OMLX_GLM_METAL_INCLUDE_ARGS)
  foreach(INCLUDE_DIR IN LISTS OMLX_GLM_METAL_INCLUDE_DIRS)
    list(APPEND OMLX_GLM_METAL_INCLUDE_ARGS -I${INCLUDE_DIR})
  endforeach()
  set(OMLX_GLM_METAL_AIR)
  foreach(SRCFILE IN LISTS OMLX_GLM_METAL_SOURCES)
    get_filename_component(TARGET_NAME ${SRCFILE} NAME_WE)
    set(AIR_FILE ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.air)
    add_custom_command(
      OUTPUT ${AIR_FILE}
      COMMAND xcrun -sdk macosx metal ${OMLX_GLM_METAL_FLAGS} -c ${SRCFILE}
              ${OMLX_GLM_METAL_INCLUDE_ARGS} -o ${AIR_FILE}
      DEPENDS ${SRCFILE} ${OMLX_GLM_METAL_HEADERS}
      COMMENT "Building ${TARGET_NAME}.air"
      VERBATIM)
    list(APPEND OMLX_GLM_METAL_AIR ${AIR_FILE})
  endforeach()

  set(OMLX_GLM_METALLIB
      ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/omlx_glm_kernels.metallib)
  add_custom_command(
    OUTPUT ${OMLX_GLM_METALLIB}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    COMMAND xcrun -sdk macosx metal ${OMLX_GLM_METAL_AIR} -o
      ${OMLX_GLM_METALLIB}
    DEPENDS ${OMLX_GLM_METAL_AIR}
    COMMENT "Building omlx_glm_kernels.metallib"
    VERBATIM)
  add_custom_target(omlx_glm_kernels_metallib DEPENDS ${OMLX_GLM_METALLIB})

  add_dependencies(omlx_glm_kernel_ops omlx_glm_kernels_metallib)
endif()

nanobind_add_module(
  _ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE omlx_glm_kernel_ops)
add_custom_command(TARGET _ext POST_BUILD
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/../smoke.py"
          "$<TARGET_FILE_DIR:_ext>"
  COMMENT "Checking MLX array ABI and evaluating native Metal scoring/top-k"
  VERBATIM)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
