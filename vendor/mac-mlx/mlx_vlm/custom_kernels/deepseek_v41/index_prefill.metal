// MLX metal_kernel body; 128 threads, one query (a one-row query tile) per TG.
// 512 retained + 1024 incoming score/id pairs: 12 KiB. FP32 matrix staging:
// Q[32,32], K[32,32], dots[32,32]: 12 KiB, plus 128 bytes of visibility.
// Total static TG memory 24,720 bytes (24.141 KiB), including tile reduction.
// Four SIMD groups cover 8 heads each, each holding four 8x8 accumulators.
// Partially sort 1024 incoming entries, keeping only their best 512. Merge
// those with the retained 512, discarding the lower half at the first stage.
// 55+10 selection stages per 1024 keys (previously 2*(45+10)); threshold-rejected
// tiles skip selection entirely. Scoring/reduction arithmetic is unchanged.
// Score ties prefer lower key ids; -inf keys retain ids until final masking.
const uint tid = thread_index_in_threadgroup;
const uint sg = tid / 32;
const uint lane = tid % 32;
const uint row = threadgroup_position_in_grid.y;
const uint batch = row / S;
const uint query = row % S;
const int limit = Lens[query];
const uint end = min(uint(N), uint(max(limit, 0)));
const uint invalid = 0x7fffffffu;
threadgroup float qs[32 * 32];
threadgroup float ks[32 * 32];
threadgroup float dots[32 * 32];
threadgroup float values[1536];
threadgroup uint ids[1536];
threadgroup uint active[32];
threadgroup float tile_max[4];
for (uint i = tid; i < 512; i += 128) {
    values[i] = -INFINITY;
    ids[i] = invalid;
}
if (!EMIT_BLOCKS && row == 0 && tid == 0) BlockMax[0] = -INFINITY;
// The causal suffix will not be visited; every output must still be written.
if (EMIT_BLOCKS) {
    const uint first_unvisited = ((end + 1023) / 1024) * (1024 / BLOCK);
    for (uint bi = first_unvisited + tid; bi < NB; bi += 128)
        BlockMax[size_t(row) * NB + bi] = -INFINITY;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint base = 0; base < end; base += 1024) {
    for (uint i = tid; i < 1024; i += 128) {
        values[512 + i] = -INFINITY;
        ids[512 + i] = base + i < N ? base + i : invalid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint sub = 0; sub < 1024 && base + sub < end; sub += 32) {
        const uint key = base + sub + lane;
        bool visible = key < N && int(key) < limit;
        if (HAS_VIS && visible) visible = Visibility[size_t(query) * N + key];
        if (HAS_CAND && visible) {
            const uint ci = CAND_BLOCKS ? key / BLOCK : key;
            visible = Candidates[size_t(row) * NC + ci];
        }
        // Each SIMD group computes the same 32-bit visibility ballot. All
        // threads take this branch together; no barrier is under a lane guard.
        const uint any_visible = simd_sum(uint(visible));
        if (sg == 0) active[lane] = uint(visible);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (any_visible != 0) {
            simdgroup_float8x8 c0(0.0f), c1(0.0f), c2(0.0f), c3(0.0f);
            for (uint db = 0; db < 128; db += 32) {
                for (uint i = tid; i < 32 * 32; i += 128) {
                    qs[i] = float(Q[(size_t(row) * 32 + i / 32) * 128 + db + i % 32]);
                    const uint ki = base + sub + i / 32;
                    // Excluded candidate/causal keys do not load from the cache.
                    ks[i] = active[i / 32] ?
                        float(Keys[(size_t(batch) * N + ki) * 128 + db + i % 32]) : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint di = 0; di < 32; di += 8) {
                    simdgroup_float8x8 a, b0, b1, b2, b3;
                    simdgroup_load(a, qs + sg * 8 * 32 + di, 32);
                    simdgroup_load(b0, ks + di, 32, ulong2(0), true);
                    simdgroup_load(b1, ks + 8 * 32 + di, 32, ulong2(0), true);
                    simdgroup_load(b2, ks + 16 * 32 + di, 32, ulong2(0), true);
                    simdgroup_load(b3, ks + 24 * 32 + di, 32, ulong2(0), true);
                    simdgroup_multiply_accumulate(c0, a, b0, c0);
                    simdgroup_multiply_accumulate(c1, a, b1, c1);
                    simdgroup_multiply_accumulate(c2, a, b2, c2);
                    simdgroup_multiply_accumulate(c3, a, b3, c3);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            simdgroup_store(c0, dots + sg * 8 * 32, 32);
            simdgroup_store(c1, dots + sg * 8 * 32 + 8, 32);
            simdgroup_store(c2, dots + sg * 8 * 32 + 16, 32);
            simdgroup_store(c3, dots + sg * 8 * 32 + 24, 32);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint ki = sg; ki < 32; ki += 4) {
                const float weighted = max(dots[lane * 32 + ki], 0.0f)
                                       * Weights[size_t(row) * 32 + lane];
                const float score = simd_sum(weighted);
                if (lane == 0 && active[ki]) values[512 + sub + ki] = score;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Compact block maxima are consumed by the existing candidate selector.
    // BLOCK divides 1024 when enabled; the final tile is -inf padded.
    if (EMIT_BLOCKS) {
        for (uint bi = tid; bi < 1024 / BLOCK; bi += 128) {
            const uint global_block = base / BLOCK + bi;
            float best = -INFINITY;
            for (uint j = 0; j < BLOCK; ++j)
                best = max(best, values[512 + bi * BLOCK + j]);
            if (global_block < NB) BlockMax[size_t(row) * NB + global_block] = best;
        }
    }
    // Do this AFTER emitting maxima: rejected top-k tiles can still contain
    // candidate-source winners. Later ids lose equal-score ties to retained ids.
    float best = -INFINITY;
    for (uint i = tid; i < 1024; i += 128) best = max(best, values[512 + i]);
    best = simd_max(best);
    if (lane == 0) tile_max[sg] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float incoming_max = max(max(tile_max[0], tile_max[1]), max(tile_max[2], tile_max[3]));
    // This threadgroup-uniform branch also handles fully masked tiles.
    if (incoming_max <= values[511]) continue;

    // Sort two incoming 512 halves in opposite directions (45 stages), then
    // merge only the winning half (10 stages). No work sorts discarded losers.
    for (uint span = 2; span <= 1024; span <<= 1) {
        for (uint stride = span >> 1; stride > 0; stride >>= 1) {
            const uint pairs = span == 1024 && stride < 512 ? 256 : 512;
            for (uint pair = tid; pair < pairs; pair += 128) {
                const uint i = (pair / stride) * (2 * stride) + pair % stride;
                const uint a = 512 + i, b = a + stride;
                const bool better = values[a] > values[b] ||
                    (values[a] == values[b] && ids[a] < ids[b]);
                const bool worse = values[a] < values[b] ||
                    (values[a] == values[b] && ids[a] > ids[b]);
                if (((i & span) == 0) ? worse : better) {
                    const float v = values[a]; values[a] = values[b]; values[b] = v;
                    const uint id = ids[a]; ids[a] = ids[b]; ids[b] = id;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    // First merge stage reads incoming winners in reverse order. Each lane
    // writes its retained slot only; incoming memory is read-only here.
    for (uint i = tid; i < 512; i += 128) {
        const uint other = 1023 - i;
        if (values[i] < values[other] || (values[i] == values[other] && ids[i] > ids[other])) {
            values[i] = values[other];
            ids[i] = ids[other];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Only the retained half needs the remaining nine merge stages.
    for (uint stride = 256; stride > 0; stride >>= 1) {
        for (uint pair = tid; pair < 256; pair += 128) {
            const uint a = (pair / stride) * (2 * stride) + pair % stride;
            const uint b = a + stride;
            if (values[a] < values[b] || (values[a] == values[b] && ids[a] > ids[b])) {
                const float v = values[a]; values[a] = values[b]; values[b] = v;
                const uint id = ids[a]; ids[a] = ids[b]; ids[b] = id;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
// Sort only the requested score winners into position order. Carry their
// scores through the sort so every -inf lane remains -1, even within lens.
for (uint i = tid; i < 512; i += 128) {
    if (i >= TOPK) { ids[i] = invalid; values[i] = -INFINITY; }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint span = 2; span <= 512; span <<= 1) {
    for (uint stride = span >> 1; stride > 0; stride >>= 1) {
        for (uint pair = tid; pair < 256; pair += 128) {
            const uint a = (pair / stride) * (2 * stride) + pair % stride;
            const uint b = a + stride;
            if (((a & span) == 0) ? ids[a] > ids[b] : ids[a] < ids[b]) {
                const uint id = ids[a]; ids[a] = ids[b]; ids[b] = id;
                const float v = values[a]; values[a] = values[b]; values[b] = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
for (uint i = tid; i < TOPK; i += 128) {
    const uint key = ids[i];
    Indices[size_t(row) * TOPK + i] = key < N && int(key) < limit && values[i] > -INFINITY
        ? int(key) + Offset[0] : -1;
}
