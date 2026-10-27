// 8 heads x 32 selected keys, four SIMD matrix tiles, fp32 throughout.
const uint tid = thread_index_in_threadgroup;
const uint sg = tid / 32;
const uint tile = threadgroup_position_in_grid.y;
const uint kt = tile % ((K + 31) / 32);
const uint ht = (tile / ((K + 31) / 32)) % ((H + 7) / 8);
const uint query = tile / (((K + 31) / 32) * ((H + 7) / 8));
const uint batch = query / M;
threadgroup float qs[8 * 64];
threadgroup float ks[32 * 64];
threadgroup float result[8 * 32];
simdgroup_float8x8 accum(0.0f);
for (uint db = 0; db < D; db += 64) {
    for (uint i = tid; i < 8 * 64; i += 128) {
        const uint head = ht * 8 + i / 64;
        const uint dim = db + i % 64;
        qs[i] = head < H && dim < D ? float(Q[(size_t(query) * H + head) * D + dim]) : 0.0f;
    }
    for (uint i = tid; i < 32 * 64; i += 128) {
        const uint slot = kt * 32 + i / 64;
        const uint dim = db + i % 64;
        const int index = slot < K ? Indices[size_t(query) * K + slot] : -1;
        float value = 0.0f;
        if (index >= 0 && index < NW + NC && dim < D)
            value = index < NW ? float(Window[(size_t(batch) * NW + index) * D + dim])
                : float(Compressed[(size_t(batch) * NC + index - NW) * D + dim]);
        ks[i] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint di = 0; di < 64; di += 8) {
        simdgroup_float8x8 a, b;
        simdgroup_load(a, qs + di, 64);
        simdgroup_load(b, ks + sg * 8 * 64 + di, 64, ulong2(0), true);
        simdgroup_multiply_accumulate(accum, a, b, accum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
simdgroup_store(accum, result + sg * 8, 32);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = tid; i < 8 * 32; i += 128) {
    const uint head = ht * 8 + i / 32;
    const uint slot = kt * 32 + i % 32;
    if (head < H && slot < K) {
        const int index = Indices[size_t(query) * K + slot];
        Scores[(size_t(query) * H + head) * K + slot] =
            index >= 0 && index < NW + NC ? result[i] * Scale : -1e30f;
    }
}
