# SwiReasoning example (VENDORED from Antonbe1b/swireasoning-llamacpp, MIT).
# See NOTICE.md / LICENSE alongside this file, and docs/SWIREASONING.md for the
# build + --swir-selftest + validation commands.
#
# NOTE: link name adapted to THIS fork's common target ("common", not the
# upstream "llama-common"). swireasoning.cpp additionally includes the internal
# header ../../src/llama-model.h to read token_embd.weight for soft embeddings.
set(TARGET llama-swir)
add_executable(${TARGET} swireasoning.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
