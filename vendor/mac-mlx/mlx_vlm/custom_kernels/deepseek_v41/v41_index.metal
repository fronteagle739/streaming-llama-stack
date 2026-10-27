// SPDX-License-Identifier: MIT
// Ported from https://github.com/IngeniousIdiocy/ds4-v41-m3ultra at c8aa547,
// metal/dsv4_misc.metal:2226-2357. See V41_KERNELS_NOTICE.md.
// Despite its GLM name, this FP32 tile is dispatched by the V4.1 batch wrapper.
const uint2 tgpig = threadgroup_position_in_grid.xy;
const uint batch = threadgroup_position_in_grid.z;
const ushort tid = thread_index_in_threadgroup;
const ushort lane = thread_index_in_simdgroup;
const ushort sg = simdgroup_index_in_threadgroup;
device const char *q = (device const char *)(Q + size_t(batch)*M*H*128);
device const char *weights = (device const char *)(Weights + size_t(batch)*M*H);
device const char *indexer_key_cache = (device const char *)(Keys + size_t(batch)*N*128);
device char *scores = (device char *)(Scores + size_t(batch)*M*N);
threadgroup float shared[8*128 + 32*128 + 8*32];

    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;

    const uint row_base = tgpig.x * TN;
    const uint token_base = tgpig.y * TM;

    threadgroup float *qtg = shared;
    threadgroup float *ktg = qtg + TM*D;
    threadgroup float *dot = ktg + TN*D;

    const uint last_token = min(token_base + TM, uint(M));
    const uint max_visible = last_token > token_base ?
        uint(Lens[last_token - 1u]) : 0u;

    if (row_base >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint tr = i / TN;
            const uint rc = i - tr*TN;
            const uint token = token_base + tr;
            const uint row = row_base + rc;
            if (token < M && row < N) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * (N*sizeof(float))) + row;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint rc = i / D;
        const uint d = i - rc*D;
        const uint row = row_base + rc;
        float v = 0.0f;
        if (row < N) {
            v = ((device const float *)indexer_key_cache)[(uint64_t)row * 128 + d];
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3;
    const uint token_row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0; head < H; head++) {
        for (uint i = tid; i < TM*D; i += 128) {
            const uint tr = i / D;
            const uint d = i - tr*D;
            const uint token = token_base + tr;
            float v = 0.0f;
            if (token < M) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * (H*128*sizeof(float)) +
                    (uint64_t)head  * (128*sizeof(float)));
                v = qrow[d];
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 mdot = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint db = 0; db < D/TS; db++) {
            simdgroup_float8x8 mq;
            simdgroup_float8x8 mk;
            simdgroup_load(mq, qtg + db*TS, D, 0, false);
            simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
            simdgroup_multiply_accumulate(mdot, mq, mk, mdot);
        }

        simdgroup_store(mdot, dot + (uint)sg * TS, TN, 0, false);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (token0 < M && row0 < N) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token0 * (H*sizeof(float)));
            const float s = dot[token_row0*TN + col0];
            acc0 += max(s * Scale[0], 0.0f) * w[head];
        }
        if (token1 < M && row1 < N) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token1 * (H*sizeof(float)));
            const float s = dot[token_row1*TN + col1];
            acc1 += max(s * Scale[0], 0.0f) * w[head];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (token0 < M && row0 < N) {
        const uint visible = uint(Lens[token0]);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * (N*sizeof(float))) + row0;
        *dst = row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < M && row1 < N) {
        const uint visible = uint(Lens[token1]);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * (N*sizeof(float))) + row1;
        *dst = row1 < visible ? acc1 : -INFINITY;
    }
