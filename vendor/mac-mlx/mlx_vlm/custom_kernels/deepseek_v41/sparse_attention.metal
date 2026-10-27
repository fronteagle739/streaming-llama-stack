// V4.1: one threadgroup per query/head, eight SIMD groups over selected rows.
// Inputs are contiguous (MLX JIT ensures this); banks may have different dtypes.
// Unlike GLM MLA, D includes rotary channels in BOTH contractions; there is no
// K-L+query causal test. Visibility is already encoded in signed Indices.
const uint tid = thread_index_in_threadgroup;
const uint lane = tid % 32;
const uint sg = tid / 32;
const uint row = threadgroup_position_in_grid.y;
const uint head = row % H;
const uint query = row / H;
const uint batch = query / M;
threadgroup float weights[640];
threadgroup float scratch[8];
threadgroup float denominator;
threadgroup float partial[8 * D];

float qv[D / 32];
for (uint j = 0; j < D / 32; ++j)
    qv[j] = float(Q[size_t(row) * D + j * 32 + lane]);

for (uint slot = sg; slot < K; slot += 8) {
    const int index = Indices[size_t(query) * K + slot];
    const bool valid = index >= 0 && index < NW + NC;
    float dot = 0.0f;
    if (valid) {
        for (uint j = 0; j < D / 32; ++j) {
            const uint dim = j * 32 + lane;
            const float key = index < NW
                ? float(Window[(size_t(batch) * NW + index) * D + dim])
                : float(Compressed[(size_t(batch) * NC + index - NW) * D + dim]);
            dot = fma(qv[j], key, dot);
        }
    }
    dot = simd_sum(dot);
    if (lane == 0) weights[slot] = valid ? dot * Scale : -1e30f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float maximum = max(float(Sink[head]), -1e30f);
for (uint slot = tid; slot < K; slot += 256)
    maximum = max(maximum, weights[slot]);
maximum = simd_max(maximum);
if (lane == 0) scratch[sg] = maximum;
threadgroup_barrier(mem_flags::mem_threadgroup);
maximum = simd_max(lane < 8 ? scratch[lane] : -1e30f);

float total = 0.0f;
for (uint slot = tid; slot < K; slot += 256) {
    const int index = Indices[size_t(query) * K + slot];
    const float w = index >= 0 && index < NW + NC
        ? exp(weights[slot] - maximum) : 0.0f;
    weights[slot] = w;
    total += w;
}
total = simd_sum(total);
// All SIMD groups finished reading scratch before it is reused.
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) scratch[sg] = total;
threadgroup_barrier(mem_flags::mem_threadgroup);
total = simd_sum(lane < 8 ? scratch[lane] : 0.0f);
if (tid == 0) denominator = total + exp(float(Sink[head]) - maximum);

float acc[D / 32];
for (uint j = 0; j < D / 32; ++j) acc[j] = 0.0f;
for (uint slot = sg; slot < K; slot += 8) {
    const int index = Indices[size_t(query) * K + slot];
    if (index < 0 || index >= NW + NC) continue;
    const float w = weights[slot];
    for (uint j = 0; j < D / 32; ++j) {
        const uint dim = j * 32 + lane;
        const float value = index < NW
            ? float(Window[(size_t(batch) * NW + index) * D + dim])
            : float(Compressed[(size_t(batch) * NC + index - NW) * D + dim]);
        acc[j] = fma(w, value, acc[j]);
    }
}
for (uint j = 0; j < D / 32; ++j)
    partial[sg * D + j * 32 + lane] = acc[j];
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint dim = tid; dim < D; dim += 256) {
    float value = 0.0f;
    for (uint group = 0; group < 8; ++group) value += partial[group * D + dim];
    Output[size_t(row) * D + dim] = T(value / denominator);
}
