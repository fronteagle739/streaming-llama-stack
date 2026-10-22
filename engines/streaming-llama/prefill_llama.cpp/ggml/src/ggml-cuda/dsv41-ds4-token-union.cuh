// Reused ds4_cuda.cu19948-20027; same algorithm/record order.
// Four-query masks need4 bits, so pack8 instead of2 masks perword.
// Original source SHA256 24c494c37c683d99d55836450d87a051db7c57d04f354da42ebbf79fd2f21660.
__global__ static void __launch_bounds__(512, 1) attention_tokentile_union_build_kernel(
        int2 *records,
        uint32_t *counts,
        const int32_t *topk,
        const int32_t *positions,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t ratio,
        uint32_t n_comp,
        uint32_t rec_stride) {
    extern __shared__ uint32_t bitmap[];
    __shared__ uint32_t scan[513];
    __shared__ uint32_t running_s;

    const uint32_t tid = threadIdx.x;
    const uint32_t tile_base = blockIdx.x * kTTTileTokens;
    if (tile_base >= n_tokens) {
        if (tid == 0u) counts[blockIdx.x] = 0u;
        return;
    }
    const uint32_t tile_count =
        n_tokens - tile_base < kTTTileTokens ? n_tokens - tile_base : kTTTileTokens;
    const uint32_t bitmap_words = (n_comp + 7u) >> 3u;

    for (uint32_t w = tid; w < bitmap_words; w += blockDim.x) {
        bitmap[w] = 0u;
    }
    if (tid == 0u) running_s = 0u;
    __syncthreads();

    const uint32_t total_slots = tile_count * top_k;
    for (uint32_t idx = tid; idx < total_slots; idx += blockDim.x) {
        const uint32_t tok = idx / top_k;
        const uint32_t i = idx - tok * top_k;
        const uint32_t t = tile_base + tok;
        const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0 + t;
        uint32_t visible = ratio ? (positions ? qpos / ratio : (qpos + 1u) / ratio) : n_comp;
        if (visible > n_comp) visible = n_comp;
        const int32_t c = topk[(uint64_t)t * top_k + i];
        if (c >= 0 && (uint32_t)c < visible) {
            const uint32_t cu = (uint32_t)c;
            const uint32_t bits = ((uint32_t)(1u << tok)) << ((cu & 7u) * 4u);
            atomicOr(&bitmap[cu >> 3u], bits);
        }
    }
    __syncthreads();

    for (uint32_t base = 0u; base < n_comp; base += blockDim.x) {
        const uint32_t id = base + tid;
        uint32_t mask = 0u;
        if (id < n_comp) {
            const uint32_t word = bitmap[id >> 3u];
            mask = (word >> ((id & 7u) * 4u)) & 0x0fu;
        }
        const uint32_t pred = mask != 0u ? 1u : 0u;
        if (tid == 0u) scan[0] = 0u;
        scan[tid + 1u] = pred;
        __syncthreads();
        for (uint32_t off = 1u; off < blockDim.x; off <<= 1u) {
            uint32_t v = 0u;
            if (tid >= off) v = scan[tid + 1u - off];
            __syncthreads();
            scan[tid + 1u] += v;
            __syncthreads();
        }
        const uint32_t rank = scan[tid];
        const uint32_t total = scan[blockDim.x];
        const uint32_t running = running_s;
        if (pred) {
            records[(uint64_t)blockIdx.x * rec_stride + running + rank] =
                make_int2((int)id, (int)mask);
        }
        __syncthreads();
        if (tid == 0u) running_s = running + total;
        __syncthreads();
    }

    if (tid == 0u) counts[blockIdx.x] = running_s;
}
