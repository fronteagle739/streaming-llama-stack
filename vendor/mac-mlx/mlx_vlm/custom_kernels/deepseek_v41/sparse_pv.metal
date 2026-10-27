// 8 heads x 32 output channels, K tiled by 32. Weights stay unnormalized
// until after fp32 PV, matching the reference's division placement.
const uint tid = thread_index_in_threadgroup;
const uint sg = tid / 32;
const uint tile = threadgroup_position_in_grid.y;
const uint dt = tile % (D / 32);
const uint ht = (tile / (D / 32)) % ((H + 7) / 8);
const uint query = tile / ((D / 32) * ((H + 7) / 8));
const uint batch = query / M;
threadgroup float ws[8 * 32];
threadgroup float vs[32 * 32];
threadgroup float result[8 * 32];
simdgroup_float8x8 accum(0.0f);
for (uint kb = 0; kb < K; kb += 32) {
    for (uint i = tid; i < 8 * 32; i += 128) {
        const uint head = ht * 8 + i / 32;
        const uint slot = kb + i % 32;
        ws[i] = head < H && slot < K ? Weights[(size_t(query) * H + head) * K + slot] : 0.0f;
    }
    for (uint i = tid; i < 32 * 32; i += 128) {
        const uint slot = kb + i / 32;
        const uint dim = dt * 32 + i % 32;
        const int index = slot < K ? Indices[size_t(query) * K + slot] : -1;
        float value = 0.0f;
        if (index >= 0 && index < NW + NC)
            value = index < NW ? float(Window[(size_t(batch) * NW + index) * D + dim])
                : float(Compressed[(size_t(batch) * NC + index - NW) * D + dim]);
        vs[i] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint ki = 0; ki < 32; ki += 8) {
        simdgroup_float8x8 a, b;
        simdgroup_load(a, ws + ki, 32);
        simdgroup_load(b, vs + ki * 32 + sg * 8, 32);
        simdgroup_multiply_accumulate(accum, a, b, accum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
simdgroup_store(accum, result + sg * 8, 32);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = tid; i < 8 * 32; i += 128) {
    const uint head = ht * 8 + i / 32;
    if (head < H) Output[(size_t(query) * H + head) * D + dt * 32 + i % 32] =
        T(result[i] / Denom[size_t(query) * H + head]);
}
