# Opt-in registration is owned by the top-level CMakeLists.txt. Reuse the owner's
# existing read-only JSON header; do not download/vendor another dependency.
if(NOT WIN32 OR NOT GGML_CUDA)
    message(FATAL_ERROR "The offline V4.1 producer requires Windows and GGML_CUDA")
endif()
if(GGML_BACKEND_DL)
    message(FATAL_ERROR "The offline V4.1 producer directly links CUDA; configure GGML_BACKEND_DL=OFF")
endif()
if(BUILD_SHARED_LIBS)
    message(FATAL_ERROR "The offline V4.1 producer binds project code in its executable SHA; configure BUILD_SHARED_LIBS=OFF")
endif()
set(LLAMA_DSV41_JSON_INCLUDE_DIR "" CACHE PATH "Existing include root containing nlohmann/json.hpp")
if(NOT EXISTS "${LLAMA_DSV41_JSON_INCLUDE_DIR}/nlohmann/json.hpp")
    message(FATAL_ERROR "Set LLAMA_DSV41_JSON_INCLUDE_DIR to the existing canonical vendor include root")
endif()
add_executable(llama-dsv41-prefill main.cpp studio-sampling.c)
set_target_properties(llama-dsv41-prefill PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
set(DSV41_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src")
foreach(unit graph plan attention ffn layer cache cache-import wire delta artifact engram engram-prefetch rope executor carry weights sweep)
    target_sources(llama-dsv41-prefill PRIVATE "${DSV41_SOURCE_DIR}/llama-dsv41-${unit}.cpp")
endforeach()
target_sources(llama-dsv41-prefill PRIVATE "${DSV41_SOURCE_DIR}/llama-weight-fetcher.cpp")
target_sources(llama-dsv41-prefill PRIVATE "${DSV41_SOURCE_DIR}/llama-kv-artifact.cpp")
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
set(LLAMA_DSV41_DSTORAGE_INCLUDE_DIR "" CACHE PATH "Existing read-only DirectStorage SDK include directory")
set(LLAMA_DSV41_DSTORAGE_LIBRARY "" CACHE FILEPATH "Existing read-only DirectStorage dstorage.lib")
if(NOT EXISTS "${LLAMA_DSV41_DSTORAGE_INCLUDE_DIR}/dstorage.h" OR NOT EXISTS "${LLAMA_DSV41_DSTORAGE_LIBRARY}")
    message(FATAL_ERROR "Point LLAMA_DSV41_DSTORAGE_INCLUDE_DIR/LIBRARY at the existing DirectStorage SDK")
endif()
target_compile_features(llama-dsv41-prefill PRIVATE cxx_std_17)
target_compile_definitions(llama-dsv41-prefill PRIVATE GGML_USE_CUDA=1 LLAMA_DIRECTSTORAGE LLAMA_BUILD)

# The delivered 1M profiles explicitly select these existing, reviewed paths.
# Ordinary engine builds retain their default-off policy.
option(LLAMA_DSV41_NATIVE_HALF_PROFILES "Enable the reviewed native-half producer profiles" OFF)
if(LLAMA_DSV41_NATIVE_HALF_PROFILES)
    target_compile_definitions(llama-dsv41-prefill PRIVATE DSV41_NATIVE_HALF_TEST=1 DSV41_NATIVE_HALF_PERF_TEST=1)
endif()
target_include_directories(llama-dsv41-prefill PRIVATE "${DSV41_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/ggml/src"
    "${LLAMA_DSV41_JSON_INCLUDE_DIR}" "${LLAMA_DSV41_DSTORAGE_INCLUDE_DIR}")
# These are the actual production units, not CPU/provider stubs. Avoid pulling
# the unrelated full llama model registry/common/chat toolchain into this CLI.
target_link_libraries(llama-dsv41-prefill PRIVATE ggml-base ggml-cpu ggml-cuda CUDA::cudart
    "${LLAMA_DSV41_DSTORAGE_LIBRARY}" ole32 d3d12 dxgi bcrypt Threads::Threads)

# Default-off diagnostics. The immutable owner manifest separately pins the
# changed capture sources; this optional value names their original base only.
target_sources(llama-dsv41-prefill PRIVATE "${DSV41_SOURCE_DIR}/boundary-capture.cpp")
set(LLAMA_DSV41_BOUNDARY_SOURCE_BASE_REVISION "" CACHE STRING "Reviewed base revision for local capture diagnostics")
if(NOT LLAMA_DSV41_BOUNDARY_SOURCE_BASE_REVISION STREQUAL "")
    target_compile_definitions(llama-dsv41-prefill PRIVATE DSV41_BOUNDARY_SOURCE_BASE_REVISION="${LLAMA_DSV41_BOUNDARY_SOURCE_BASE_REVISION}")
endif()
