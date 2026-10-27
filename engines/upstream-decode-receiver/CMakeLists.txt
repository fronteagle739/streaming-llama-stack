set(TARGET llama-upstream-decode-receiver)
if(NOT COSMOS_UPSTREAM_RECEIVER_SOURCE)
    message(FATAL_ERROR "Set COSMOS_UPSTREAM_RECEIVER_SOURCE to the retained streaming-llama source")
endif()
set(RECEIVER_ROOT "${COSMOS_UPSTREAM_RECEIVER_SOURCE}")
add_executable(${TARGET}
    "${RECEIVER_ROOT}/tools/decode-receiver/main.cpp"
    "${RECEIVER_ROOT}/tools/decode-receiver/tail_tokens.cpp"
    "${RECEIVER_ROOT}/src/llama-kv-artifact.cpp"
    "${RECEIVER_ROOT}/src/llama-kv-receiver.cpp"
    "${RECEIVER_ROOT}/src/llama-token-stream.cpp"
    "${RECEIVER_ROOT}/src/llama-tb-transport.cpp"
    wir-disabled.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
if(WIN32)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif()
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE COSMOS_UPSTREAM_RECEIVER=1)
target_include_directories(${TARGET} PRIVATE
    "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/common"
    "${RECEIVER_ROOT}/src" "${RECEIVER_ROOT}/tools/decode-receiver")
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
