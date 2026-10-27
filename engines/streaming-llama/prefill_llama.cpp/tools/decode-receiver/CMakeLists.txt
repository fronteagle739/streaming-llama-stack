set(TARGET llama-decode-receiver)
add_executable(${TARGET}
    main.cpp
    tail_tokens.cpp
    ../../src/llama-kv-artifact.cpp
    ../../src/llama-kv-receiver.cpp
    ../../src/llama-token-stream.cpp
    ../../src/llama-wir.cpp
    # PROMPT-DIRECT KV-handshake (Gate #2): control socket + hash helpers
    ../../src/llama-tb-transport.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
if(WIN32)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif()
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE ../../src)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
