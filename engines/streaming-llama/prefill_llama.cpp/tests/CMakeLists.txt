llama_add_compile_flags()

function(llama_build source)
    set(TEST_SOURCES ${source} ${ARGN})

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE common)
    install(TARGETS ${TEST_TARGET} RUNTIME)
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_test_cmd target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS} get-model.cpp)

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    target_link_libraries(${TEST_TARGET} PRIVATE common)

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
endif ()

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-sampling.cpp)
    llama_build_and_test(test-grammar-parser.cpp)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp)
    llama_build_and_test(test-chat.cpp)
    # TODO: disabled on loongarch64 because the ggml-ci node lacks Python 3.8
    if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
        llama_build_and_test(test-json-schema-to-grammar.cpp   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        target_include_directories(test-json-schema-to-grammar PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    endif()

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW
endif()

llama_build_and_test(test-chat-parser.cpp)
llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-chat-template.cpp)
llama_build_and_test(test-jinja.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
llama_build_and_test(test-json-partial.cpp)
llama_build_and_test(test-log.cpp)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-json-serialization.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
)
llama_build_and_test(test-regex-partial.cpp)

# Wavefront convergence test (build only — requires model paths at runtime)
llama_build(test-wavefront-convergence.cpp)

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    set(MODEL_NAME "tinyllamas/stories15M-q4_0.gguf")
    set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
    set(MODEL_NAME "tinyllamas/stories15M-be.Q4_0.gguf")
    set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")

add_test(NAME test-download-model COMMAND ${CMAKE_COMMAND}
    -DDEST=${MODEL_DEST}
    -DNAME=${MODEL_NAME}
    -DHASH=${MODEL_HASH}
    -P ${CMAKE_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

llama_build_and_test(test-thread-safety.cpp ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-arg-parser.cpp)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp)
endif()
llama_build_and_test(test-gguf.cpp)
llama_build_and_test(test-backend-ops.cpp)

llama_build_and_test(test-model-load-cancel.cpp LABEL "model")
llama_build_and_test(test-autorelease.cpp       LABEL "model")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model")

# Test for state restore with fragmented KV cache
# Requires a model, uses same args pattern as test-thread-safety
llama_build_and_test(test-state-restore-fragmented.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-state-restore-fragmented PROPERTIES FIXTURES_REQUIRED test-download-model)

if (NOT GGML_BACKEND_DL)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp)
    llama_build_and_test(test-quantize-fns.cpp)
    llama_build_and_test(test-quantize-perf.cpp)
    llama_build_and_test(test-rope.cpp)
endif()

# libmtmd
set(LLAMA_TEST_NAME test-mtmd-c-api)
llama_build_and_test(test-mtmd-c-api.c)
target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

llama_build_and_test(test-alloc.cpp)
target_include_directories(test-alloc PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)

# Reset LLAMA_TEST_NAME from prior test definitions
unset(LLAMA_TEST_NAME)

# Token stream unit tests (no model required)
llama_build(test-token-stream.cpp ../src/llama-token-stream.cpp)
target_include_directories(test-token-stream PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test-token-stream PRIVATE ${CMAKE_THREAD_LIBS_INIT})
llama_test(test-token-stream)

# Current-upstream GLM DSA state-wire conversion tests (no model required)
add_executable(test-upstream-dsa-wire test-upstream-dsa-wire.cpp ../src/llama-upstream-dsa-wire.cpp)
target_include_directories(test-upstream-dsa-wire PRIVATE ${PROJECT_SOURCE_DIR}/src)
llama_test(test-upstream-dsa-wire)

# KV import scatter map unit tests (no model required)
llama_build(test-kv-import.cpp ../src/llama-kv-import.cpp)
target_include_directories(test-kv-import PRIVATE ${PROJECT_SOURCE_DIR}/src)
llama_test(test-kv-import)

# KV artifact header v1.1 round-trip tests (no model required)
llama_build(test-kv-artifact.cpp ../src/llama-kv-artifact.cpp)
target_include_directories(test-kv-artifact PRIVATE ${PROJECT_SOURCE_DIR}/src)
llama_test(test-kv-artifact)

# Persistent KV-artifact store (Engine A / Phases A1+A2) tests (no model required)
llama_build(test-kv-store.cpp ../src/llama-kv-store.cpp ../src/llama-kv-artifact.cpp)
target_include_directories(test-kv-store PRIVATE ${PROJECT_SOURCE_DIR}/src)
llama_test(test-kv-store)

# Prefill strategy selector tests (no model required)
llama_build(test-prefill-strategy.cpp ../src/llama-prefill-strategy.cpp)
target_include_directories(test-prefill-strategy PRIVATE ${PROJECT_SOURCE_DIR}/src)
llama_test(test-prefill-strategy)

# Retained-route capability policy tests (no model or backend required)
llama_build(test-retained-route.cpp)
target_include_directories(test-retained-route PRIVATE ${PROJECT_SOURCE_DIR}/src)
llama_test(test-retained-route)

# Streaming local decode selector tests (no model required)
llama_build(test-streaming-local-decode.cpp ../src/llama-streaming-local-decode.cpp ../src/llama-mtp-policy.cpp)
target_include_directories(test-streaming-local-decode PRIVATE ${PROJECT_SOURCE_DIR}/src)
llama_test(test-streaming-local-decode)

# Speculative rejection-sampling helpers (no model required)
llama_build(test-speculative-sampling.cpp)
llama_test(test-speculative-sampling)

# GLM/NextN MTP policy tests (no model required)
llama_build(test-mtp-policy.cpp ../src/llama-mtp-policy.cpp)
target_include_directories(test-mtp-policy PRIVATE ${PROJECT_SOURCE_DIR}/src)
llama_test(test-mtp-policy)

# KV receiver loopback tests (no model required, uses TB transport sender)
llama_build(test-kv-receiver.cpp ../src/llama-kv-receiver.cpp ../src/llama-kv-artifact.cpp ../src/llama-tb-transport.cpp)
target_include_directories(test-kv-receiver PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(WIN32)
    target_link_libraries(test-kv-receiver PRIVATE ws2_32)
endif()
llama_test(test-kv-receiver)

# Layer-major ubatch processing tests (no model required for unit tests)
llama_build(test-layer-major.cpp)
target_include_directories(test-layer-major PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/ggml/src)
llama_test(test-layer-major)

# DSV4 packed expert-slab index tests (CPU only; optional header-only real GGUF check)
llama_build(test-expert-index.cpp ../src/llama-expert-index-arithmetic.cpp)
target_include_directories(test-expert-index PRIVATE ${PROJECT_SOURCE_DIR}/src)
llama_test(test-expert-index)

# DSV4 streaming-decode abstract expert-slot LRU policy tests (CPU only)
llama_build(test-expert-lru-cache.cpp ../src/llama-expert-lru-cache.cpp)
target_include_directories(test-expert-lru-cache PRIVATE ${PROJECT_SOURCE_DIR}/src)
set(LLAMA_EXPERT_LRU_TRACE_ROOT "" CACHE PATH "Directory containing expert LRU oracle traces")
llama_test(test-expert-lru-cache ARGS --trace-root "${LLAMA_EXPERT_LRU_TRACE_ROOT}")

# WF buffer padding test (CUDA required, no model)
if (GGML_CUDA)
    llama_build(test-wf-padding.cpp)
    target_include_directories(test-wf-padding PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
    llama_test(test-wf-padding)
endif()

# Weight fetcher unit tests (no model required)
llama_build(test-weight-fetcher.cpp)
target_include_directories(test-weight-fetcher PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test-weight-fetcher PRIVATE ${CMAKE_THREAD_LIBS_INIT})
llama_test(test-weight-fetcher)

# DirectStorage support for weight fetcher benchmark (optional, Windows only)
set(DS_SDK_DIR "${PROJECT_SOURCE_DIR}/external/directstorage" CACHE PATH "DirectStorage SDK package root")
if (WIN32 AND EXISTS "${DS_SDK_DIR}/native/include/dstorage.h")
    message(STATUS "DirectStorage SDK found at ${DS_SDK_DIR} — enabling --dstorage + --dstorage-gpu benchmarks")
    target_include_directories(test-weight-fetcher PRIVATE "${DS_SDK_DIR}/native/include")
    target_link_libraries(test-weight-fetcher PRIVATE
        "${DS_SDK_DIR}/native/lib/x64/dstorage.lib"
        ole32
        d3d12
        dxgi)
    target_compile_definitions(test-weight-fetcher PRIVATE LLAMA_DIRECTSTORAGE)
    # CUDA interop (D3D12 → CUDA) — needed for --dstorage-cuda test
    if (GGML_CUDA)
        find_package(CUDAToolkit)
        if (CUDAToolkit_FOUND)
            target_link_libraries(test-weight-fetcher PRIVATE CUDA::cudart)
            target_compile_definitions(test-weight-fetcher PRIVATE LLAMA_CUDA_INTEROP)
            message(STATUS "CUDA interop enabled for test-weight-fetcher (--dstorage-cuda)")
        endif()
    endif()
    # Copy runtime DLLs beside the test binary (if present in SDK)
    if (EXISTS "${DS_SDK_DIR}/native/bin/x64/dstorage.dll")
        add_custom_command(TARGET test-weight-fetcher POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${DS_SDK_DIR}/native/bin/x64/dstorage.dll"
                $<TARGET_FILE_DIR:test-weight-fetcher>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${DS_SDK_DIR}/native/bin/x64/dstoragecore.dll"
                $<TARGET_FILE_DIR:test-weight-fetcher>
            COMMENT "Copying DirectStorage DLLs to output directory"
        )
    endif()
endif()

# DirectStorage chunked-read test (standalone, no llama dependency)
if (WIN32 AND EXISTS "${DS_SDK_DIR}/native/include/dstorage.h" AND GGML_CUDA)
    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        add_executable(test-ds-chunks test-ds-chunks.cpp)
        target_include_directories(test-ds-chunks PRIVATE "${DS_SDK_DIR}/native/include")
        target_link_libraries(test-ds-chunks PRIVATE
            "${DS_SDK_DIR}/native/lib/x64/dstorage.lib"
            ole32 d3d12 dxgi CUDA::cudart)
        target_compile_definitions(test-ds-chunks PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
        # Copy runtime DLLs
        if (EXISTS "${DS_SDK_DIR}/native/bin/x64/dstorage.dll")
            add_custom_command(TARGET test-ds-chunks POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DS_SDK_DIR}/native/bin/x64/dstorage.dll"
                    $<TARGET_FILE_DIR:test-ds-chunks>
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DS_SDK_DIR}/native/bin/x64/dstoragecore.dll"
                    $<TARGET_FILE_DIR:test-ds-chunks>
            )
        endif()
        message(STATUS "test-ds-chunks enabled (DirectStorage + CUDA chunked read test)")
    endif()
endif()

# E2E network handoff test (requires model, pass -m on command line)
# Compiles transport/receiver/stream sources directly (not via llama.dll) to avoid export issues
llama_build(test-e2e-network-handoff.cpp
    ../src/llama-kv-artifact.cpp
    ../src/llama-kv-receiver.cpp
    ../src/llama-token-stream.cpp
    ../src/llama-tb-transport.cpp)
target_include_directories(test-e2e-network-handoff PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(WIN32)
    target_link_libraries(test-e2e-network-handoff PRIVATE ws2_32)
endif()

# E2E handoff integration test (requires model)
# Compile llama-token-stream.cpp directly (NOT via llama.dll, which does
# not export llama_token_stream_server/client) + link ws2_32 on Windows — mirrors
# test-e2e-network-handoff above. Without this the target fails to link (LNK2019 on
# llama_token_stream_server::{start,enqueue,...} / client::{connect,read_msg,...}).
llama_build_and_test(test-e2e-handoff.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
target_sources(test-e2e-handoff PRIVATE ${PROJECT_SOURCE_DIR}/src/llama-token-stream.cpp)
target_include_directories(test-e2e-handoff PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test-e2e-handoff PRIVATE ${CMAKE_THREAD_LIBS_INIT})
if(WIN32)
    target_link_libraries(test-e2e-handoff PRIVATE ws2_32)
endif()
set_tests_properties(test-e2e-handoff PROPERTIES FIXTURES_REQUIRED test-download-model)
