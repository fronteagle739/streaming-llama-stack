// SPDX-License-Identifier: MIT
// Ported from https://github.com/IngeniousIdiocy/ds4-v41-m3ultra at c8aa547,
// metal/dsv4_misc.metal:5784-5882,6116-6286. See V41_KERNELS_NOTICE.md.
// Arithmetic helpers are unchanged. Bank addressing and ABI are adapted.
static inline void dsv4_attend_shared_h4_row(
        threadgroup const half4 *kv4,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    const half4 k0 = kv4[lane +  0];
    const half4 k1 = kv4[lane + 32];
    const half4 k2 = kv4[lane + 64];
    const half4 k3 = kv4[lane + 96];

    float score = dot((float4)q0, (float4)k0) +
                  dot((float4)q1, (float4)k1) +
                  dot((float4)q2, (float4)k2) +
                  dot((float4)q3, (float4)k3);
    score = simd_sum(score) * scale;

    const float old_m = M;
    const float new_m = max(M, score);
    const float old_scale = exp(old_m - new_m);
    const float row_scale = exp(score - new_m);

    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;

    o0 += (float4)k0 * row_scale;
    o1 += (float4)k1 * row_scale;
    o2 += (float4)k2 * row_scale;
    o3 += (float4)k3 * row_scale;
    M = new_m;
}

static inline void dsv4_attend_shared_h4_row_at(
        threadgroup const half4 *kv4,
        uint row_in_tg,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    dsv4_attend_shared_h4_row(kv4 + row_in_tg * 128u,
                              q0, q1, q2, q3,
                              scale,
                              lane,
                              M, S,
                              o0, o1, o2, o3);
}

static inline void dsv4_attend_sink(
        float score,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    const float old_m = M;
    const float new_m = max(M, score);
    const float old_scale = exp(old_m - new_m);
    const float row_scale = exp(score - new_m);

    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;
    M = new_m;
}

static inline uint dsv4_nth_set_bit(ushort mask, uint n) {
    // The n-th set bit of mask, counting from the least significant.  Only
    // reached when the sixteen-entry window contains an absent or invisible
    // id; with a full window the caller uses the identity mapping.
    for (uint j = 0; j < 16u; j++) {
        if (mask & (ushort)(1u << j)) {
            if (n == 0u) return j;
            n--;
        }
    }
    return 15u;
}

// MLX_HEADER_END

const uint2 tgpig = threadgroup_position_in_grid.xy;
const uint batch = threadgroup_position_in_grid.z;
const ushort tid = thread_index_in_threadgroup;
const ushort lane = thread_index_in_simdgroup;
const ushort sg = simdgroup_index_in_threadgroup;
threadgroup half4 kv_shared[BLOCK * 128];

    const uint token = tgpig.x;
    const uint head = tgpig.y * (uint)COHORT + (uint)sg;
    if (token >= NT || head >= H) {
        return;
    }

    device const half4 *q4 = (device const half4 *)(Q +
        ((size_t(batch)*NT + token)*H + head)*512);
    const half4 q0 = (half4)q4[lane +  0];
    const half4 q1 = (half4)q4[lane + 32];
    const half4 q2 = (half4)q4[lane + 64];
    const half4 q3 = (half4)q4[lane + 96];

    float M = -FLT_MAX/2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    // Local IDs already encode visibility and both banks; retain slot order.
    device const int32_t *row_topk = Indices + (size_t(batch)*NT + token)*K;
    bool stop = false;
    for (uint i = 0; i < K && !stop; i += (uint)BLOCK) {
        uint rows[LEAN ? 1 : BLOCK];
        ushort mask = 0;
        uint n_rows = 0;
        for (uint j = 0; j < (uint)BLOCK && i + j < K; j++) {
            const int32_t idx = row_topk[i + j];
            if (idx < 0) {
                continue;
            }
            // Unlike the upstream sorted-ID early break, skip invalid slots:
            // our later slots may still be visible (including another bank).
            if ((uint)idx >= NW + NC) continue;
            if (LEAN) {
                mask |= (ushort)(1u << j);
            } else {
                rows[n_rows] = (uint)idx;
            }
            n_rows++;
        }
        if (n_rows == 0) {
            continue;
        }
        const bool dense = LEAN && mask == (ushort)((1u << (uint)BLOCK) - 1u);
        for (uint off = (uint)tid; off < n_rows * 128u; off += 32u * (uint)COHORT) {
            const uint r = off >> 7;
            const uint c = off & 127u;
            uint src_row;
            if (LEAN) {
                /* The r-th set bit of mask is exactly the entry the wave-2
                 * kernel stored at rows[r], so the rows and their order are
                 * unchanged. */
                const uint j = dense ? r : dsv4_nth_set_bit(mask, r);
                src_row = (uint)row_topk[i + j];
            } else {
                src_row = rows[r];
            }
            device const half *row = src_row < NW
                ? Window + (size_t(batch)*NW + src_row)*512
                : Compressed + (size_t(batch)*NC + src_row - NW)*512;
            kv_shared[off] = ((device const half4 *)row)[c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < n_rows; r++) {
            dsv4_attend_shared_h4_row_at(kv_shared,
                                         r,
                                         q0, q1, q2, q3,
                                         Scale[0],
                                         lane,
                                         M, S,
                                         o0, o1, o2, o3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    dsv4_attend_sink(Sink[head], M, S, o0, o1, o2, o3);

    const float inv_s = S == 0.0f ? 0.0f : 1.0f/S;
    device float4 *dst4 = (device float4 *)(Output +
        ((size_t(batch)*NT + token)*H + head)*512);
    dst4[lane +  0] = o0 * inv_s;
    dst4[lane + 32] = o1 * inv_s;
    dst4[lane + 64] = o2 * inv_s;
    dst4[lane + 96] = o3 * inv_s;
