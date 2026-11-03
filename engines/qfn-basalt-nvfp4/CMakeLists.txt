cmake_minimum_required(VERSION 3.24)
project(qfn_basalt_nvfp4 LANGUAGES CXX CUDA)
if(NOT WIN32)
  message(FATAL_ERROR "The QFN kernel bridge targets native Windows CUDA")
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
  message(FATAL_ERROR "NVFP4 requires CUDA 13 or later")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
find_package(CUDAToolkit REQUIRED)
set(vendor "${CMAKE_CURRENT_SOURCE_DIR}/vendor/basalt")
add_library(qfn-basalt-nvfp4 SHARED bridge.cu qsa-indexer.cu mmq-bridge.cu calibrated-a4.cu basalt-attention.cu
  "${vendor}/src/prefill/mmq/instances/mmq_nvfp4.cu"
  "${vendor}/src/prefill/mmq/quantize.cu"
  "${vendor}/src/prefill/mmq/mmq_host.cu"
  "${vendor}/src/prefill/fp4/moe_fp4.cu"
  "${vendor}/src/kernels/cuda/iq_kernels.cu")
set_target_properties(qfn-basalt-nvfp4 PROPERTIES CUDA_ARCHITECTURES 120a-real
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../bin")
target_compile_definitions(qfn-basalt-nvfp4 PRIVATE QFN_NVFP4_BUILD)
target_include_directories(qfn-basalt-nvfp4 PRIVATE "${vendor}/include")
target_compile_options(qfn-basalt-nvfp4 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor,-Zc:__cplusplus,/bigobj>)
set_source_files_properties(mmq-bridge.cu
  "${vendor}/src/prefill/mmq/instances/mmq_nvfp4.cu"
  "${vendor}/src/prefill/mmq/quantize.cu"
  "${vendor}/src/prefill/mmq/mmq_host.cu"
  PROPERTIES COMPILE_OPTIONS "--extended-lambda;-use_fast_math")
target_link_libraries(qfn-basalt-nvfp4 PRIVATE CUDA::cudart)
