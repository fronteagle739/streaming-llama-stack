// Four independent queries per threadgroup; one 32-lane SIMD group per query,
// with eight heads per grid tile. All four query states are live concurrently.
// GATHERED=true consumes one shared pre-gathered K=V buffer across head tiles;
// false is the single-launch comparison, repeating sparse bank loads per tile.
// Eight keys per block, QK = 64 depth-8 MMAs, PV = 64 width-8 MMAs.
// 64 output fragments = 128 FP32 values/lane (not 1024 for all 64 heads).
// Storage: 4*(8*32 + 8*32 + 8*8 + 8*8)*4 = 10,240 bytes.
// No full KV tile: expand exact BF16 values to FP32 in 32-channel slices.
// Full BF16 staging alone would take 4*8*512*2 = 32 KiB; 16 rows = 64 KiB.
// Scratch regions and barriers are SIMD-private; no cross-query dependency.
const uint tid = thread_index_in_threadgroup;
const uint sg = tid / 32;
const uint lane = tid % 32;
const uint tile = threadgroup_position_in_grid.y;
const uint ht = tile % ((H + 7) / 8);
const uint qt = (tile / ((H + 7) / 8)) % ((M + 3) / 4);
const uint batch = tile / (((H + 7) / 8) * ((M + 3) / 4));
const uint qi = qt * 4 + sg;
// SIMD-uniform return is safe: every barrier below has SIMD scope only.
if (qi >= M) return;
const size_t query = size_t(batch) * M + qi;

threadgroup float q_storage[4 * 8 * 32];
threadgroup float x_storage[4 * 8 * 32];
threadgroup float w_storage[4 * 8 * 8];
threadgroup float r_storage[4 * 8 * 8];
threadgroup float* qs = q_storage + sg * 8 * 32;
threadgroup float* xs = x_storage + sg * 8 * 32;
threadgroup float* weights = w_storage + sg * 8 * 8;
threadgroup float* rescales = r_storage + sg * 8 * 8;

simdgroup_float8x8 output[D / 8];
#pragma clang loop unroll(full)
for (uint dt = 0; dt < D / 8; ++dt)
    output[dt] = simdgroup_float8x8(0.0f);
// Lanes 0..7 each own one head's online softmax state, in registers.
const uint head = ht * 8 + lane;
const float sink = lane < 8 && head < H ? float(Sink[head]) : 0.0f;
float maximum = max(sink, -1e30f);
float denominator = exp(sink - maximum);

for (uint kb = 0; kb < uint(K); kb += 8) {
    const int own_id = lane < 8 && kb + lane < uint(K)
        ? Indices[query * K + kb + lane] : -1;
    const uint bit = lane < 8 && own_id >= 0 && own_id < NW + NC
        ? (1u << lane) : 0u;
    const uint valid = simd_sum(bit);
    if (valid == 0) continue;  // SIMD-uniform, including completely masked blocks.

    simdgroup_float8x8 score(0.0f);
    for (uint db = 0; db < D; db += 32) {
        for (uint i = lane; i < 8 * 32; i += 32) {
            const uint row = i / 32;
            const uint dim = db + i % 32;
            const uint h = ht * 8 + row;
            qs[i] = h < H ? float(Q[(query * H + h) * D + dim]) : 0.0f;
            float value = 0.0f;
            if (valid & (1u << row)) {
                if (GATHERED) {
                    value = float(Window[(query * K + kb + row) * D + dim]);
                } else {
                    const int id = Indices[query * K + kb + row];
                    value = id < NW ? float(Window[(size_t(batch) * NW + id) * D + dim])
                        : float(Compressed[(size_t(batch) * NC + id - NW) * D + dim]);
                }
            }
            xs[i] = value;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint di = 0; di < 32; di += 8) {
            simdgroup_float8x8 a, b;
            simdgroup_load(a, qs + di, 32);
            simdgroup_load(b, xs + di, 32, ulong2(0), true);
            simdgroup_multiply_accumulate(score, a, b, score);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(score, weights, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 8) {
        float next_max = maximum;
        for (uint r = 0; r < 8; ++r)
            if (valid & (1u << r))
                next_max = max(next_max, weights[lane * 8 + r] * Scale);
        const float alpha = exp(maximum - next_max);
        float total = denominator * alpha;
        for (uint r = 0; r < 8; ++r) {
            const float w = (valid & (1u << r))
                ? exp(weights[lane * 8 + r] * Scale - next_max) : 0.0f;
            weights[lane * 8 + r] = w;
            rescales[lane * 8 + r] = alpha;
            total += w;
        }
        maximum = next_max;
        denominator = total;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 rescale, probability;
    simdgroup_load(rescale, rescales, 8);
    simdgroup_load(probability, weights, 8);
    #pragma clang loop unroll(full)
    for (uint db = 0; db < D; db += 32) {
        for (uint i = lane; i < 8 * 32; i += 32) {
            const uint row = i / 32;
            const uint dim = db + i % 32;
            float value = 0.0f;
            if (valid & (1u << row)) {
                if (GATHERED) {
                    value = float(Window[(query * K + kb + row) * D + dim]);
                } else {
                    const int id = Indices[query * K + kb + row];
                    value = id < NW ? float(Window[(size_t(batch) * NW + id) * D + dim])
                        : float(Compressed[(size_t(batch) * NC + id - NW) * D + dim]);
                }
            }
            xs[i] = value;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint di = 0; di < 32; di += 8) {
            // Both matrices have the same layout. Elementwise multiplication
            // needs no assumption about which lane owns which matrix row.
            #pragma clang loop unroll(full)
            for (uint e = 0; e < 2; ++e)
                output[(db + di) / 8].thread_elements()[e] *= rescale.thread_elements()[e];
            simdgroup_float8x8 value;
            simdgroup_load(value, xs + di, 32);
            simdgroup_multiply_accumulate(output[(db + di) / 8], probability,
                                         value, output[(db + di) / 8]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Sink was inserted exactly once in the initial denominator, with zero value.
// Reuse Q scratch for stores; head lanes broadcast their denominator via a
// matrix of row-wise reciprocals, preserving the matrix's opaque lane layout.
if (lane < 8)
    for (uint r = 0; r < 8; ++r)
        rescales[lane * 8 + r] = 1.0f / denominator;
simdgroup_barrier(mem_flags::mem_threadgroup);
simdgroup_float8x8 inverse_sum;
simdgroup_load(inverse_sum, rescales, 8);
#pragma clang loop unroll(full)
for (uint db = 0; db < D; db += 32) {
    #pragma clang loop unroll(full)
    for (uint di = 0; di < 32; di += 8) {
        #pragma clang loop unroll(full)
        for (uint e = 0; e < 2; ++e)
            output[(db + di) / 8].thread_elements()[e] *= inverse_sum.thread_elements()[e];
        simdgroup_store(output[(db + di) / 8], qs + di, 32);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < 8 * 32; i += 32) {
        const uint h = ht * 8 + i / 32;
        if (h < H)
            Output[(query * H + h) * D + db + i % 32] = T(qs[i]);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}
