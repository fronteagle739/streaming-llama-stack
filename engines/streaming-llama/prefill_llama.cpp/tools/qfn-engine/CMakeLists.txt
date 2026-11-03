# QFNSTREAM: Qwen3.8 Flash Next streaming engine (opt-in, LLAMA_QFN_ENGINE=ON).
if(NOT WIN32 OR NOT GGML_CUDA)
    message(FATAL_ERROR "The QFN streaming engine requires Windows and GGML_CUDA")
endif()
find_package(CUDAToolkit REQUIRED)

set(QFN_SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(QFN_JSON_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/vendor" CACHE PATH "Include root containing nlohmann/json.hpp")
set(QFN_DSTORAGE_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external/directstorage/native/include" CACHE PATH "DirectStorage SDK include directory")
set(QFN_DSTORAGE_LIBRARY "${PROJECT_SOURCE_DIR}/external/directstorage/native/lib/x64/dstorage.lib" CACHE FILEPATH "DirectStorage dstorage.lib")
if(NOT EXISTS "${QFN_JSON_INCLUDE_DIR}/nlohmann/json.hpp")
    message(FATAL_ERROR "QFN_JSON_INCLUDE_DIR must contain nlohmann/json.hpp")
endif()
if(NOT EXISTS "${QFN_DSTORAGE_INCLUDE_DIR}/dstorage.h" OR NOT EXISTS "${QFN_DSTORAGE_LIBRARY}")
    message(FATAL_ERROR "Point QFN_DSTORAGE_INCLUDE_DIR/LIBRARY at the DirectStorage SDK")
endif()

# Op checks: every new ggml op/type on CUDA against the CPU reference.
add_executable(qfn-op-test qfn-op-test.cpp)
target_compile_features(qfn-op-test PRIVATE cxx_std_17)
target_include_directories(qfn-op-test PRIVATE "${PROJECT_SOURCE_DIR}/ggml/src" "${QFN_DSTORAGE_INCLUDE_DIR}")
target_link_libraries(qfn-op-test PRIVATE ggml-base ggml-cpu ggml-cuda ggml CUDA::cudart "${QFN_DSTORAGE_LIBRARY}" d3d12 dxgi ole32 Threads::Threads)

# The engine: ggml + the stack's DirectStorage weight fetcher + its IoRing Engram row loader.
add_executable(qfn-engine qfn-main.cpp qfn-serve.cpp qfn-engine.cpp qfn-pack.cpp qfn-tokenizer.cpp
    "${QFN_SRC_DIR}/llama-weight-fetcher.cpp"
    "${QFN_SRC_DIR}/llama-dsv41-engram-prefetch.cpp"
    "${QFN_SRC_DIR}/llama-dsv41-engram.cpp")
target_compile_features(qfn-engine PRIVATE cxx_std_17)
target_compile_definitions(qfn-engine PRIVATE GGML_USE_CUDA=1 LLAMA_DIRECTSTORAGE LLAMA_BUILD WF_STATIC)
target_include_directories(qfn-engine PRIVATE "${QFN_SRC_DIR}" "${PROJECT_SOURCE_DIR}/ggml/src" "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/common" "${QFN_JSON_INCLUDE_DIR}" "${QFN_DSTORAGE_INCLUDE_DIR}")
target_link_libraries(qfn-engine PRIVATE common llama ggml-base ggml-cpu ggml-cuda ggml CUDA::cudart
    "${QFN_DSTORAGE_LIBRARY}" Normaliz ole32 d3d12 dxgi bcrypt ws2_32 Threads::Threads)

# One adjacent, explicitly checked C ABI DLL. Basalt's private quantization types never enter ggml's enum.
get_filename_component(QFN_STACK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
if(NOT TARGET qfn-basalt-nvfp4)
    add_subdirectory("${QFN_STACK_ROOT}/engines/qfn-basalt-nvfp4" "${CMAKE_CURRENT_BINARY_DIR}/basalt-nvfp4")
endif()
add_dependencies(qfn-engine qfn-basalt-nvfp4)
add_dependencies(qfn-op-test qfn-basalt-nvfp4)
set_target_properties(qfn-engine qfn-op-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${QFN_STACK_ROOT}/bin")
