// SPDX-License-Identifier: MIT
// Explicit instantiation of the MMQ kernel for NVFP4 (Basalt's tile loader, llama.cpp's per-16-scale dot); the
// pattern of llama.cpp commit 3cf03257f219afbe7334045ff7c6a06ac68c627d, ggml/src/ggml-cuda/template-instances.
// Copyright (c) 2023-2026 The ggml authors.  MIT licence: see THIRD_PARTY_LICENSES.
#include "../mmq.cuh"

DECL_MMQ_CASE(GGML_TYPE_NVFP4);
