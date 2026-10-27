// Each SIMD group streams disjoint K rows, reusing its key registers for PV.
// Output slices expose independent groups without a second launch. The final
// merge reuses SG*32 floats, never SG*D floats. All contractions include RoPE.
const uint tid = thread_index_in_threadgroup;
const uint lane = tid % 32;
const uint sg = tid / 32;
const uint row = threadgroup_position_in_grid.y;
const uint slice = threadgroup_position_in_grid.z;
const uint head = row % H;
const uint query = row / H;
const uint batch = query / M;
constexpr uint GROUPS = THREADS / 32;
threadgroup float scratch[GROUPS * 32];
threadgroup float maxima[GROUPS];
threadgroup float sums[GROUPS];
float qv[D / 32];
float acc[D / SLICES / 32];
for (uint j = 0; j < D / 32; ++j)
    qv[j] = float(Q[size_t(row) * D + j * 32 + lane]);
for (uint j = 0; j < D / SLICES / 32; ++j) acc[j] = 0;
// Include the sink once in the merge, not once per SIMD group.
float maximum = max(float(Sink[head]), -1e30f);
float denominator = 0;
for (uint slot = sg; slot < K; slot += GROUPS) {
    const int index = Indices[size_t(query) * K + slot];
    if (index < 0 || index >= NW + NC) continue;
    float kv[D / 32];
    float dot = 0;
    for (uint j = 0; j < D / 32; ++j) {
        const uint dim = j * 32 + lane;
        kv[j] = index < NW
            ? float(Window[(size_t(batch) * NW + index) * D + dim])
            : float(Compressed[(size_t(batch) * NC + index - NW) * D + dim]);
        dot = fma(qv[j], kv[j], dot);
    }
    dot = simd_sum(dot) * Scale;
    const float next_max = max(maximum, dot);
    const float correction = exp(maximum - next_max);
    const float weight = exp(dot - next_max);
    denominator = denominator * correction + weight;
    for (uint j = 0; j < D / SLICES / 32; ++j)
        acc[j] = fma(weight, kv[slice * (D / SLICES / 32) + j], acc[j] * correction);
    maximum = next_max;
}
if (lane == 0) { maxima[sg] = maximum; sums[sg] = denominator; }
threadgroup_barrier(mem_flags::mem_threadgroup);
const float global_max = simd_max(lane < GROUPS ? maxima[lane] : -INFINITY);
const float total = simd_sum(lane < GROUPS ? sums[lane] * exp(maxima[lane] - global_max) : 0.0f)
    + exp(float(Sink[head]) - global_max);
const float factor = exp(maximum - global_max) / total;
for (uint j = 0; j < D / SLICES / 32; ++j) {
    scratch[sg * 32 + lane] = acc[j] * factor;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float value = 0;
        for (uint group = 0; group < GROUPS; ++group) value += scratch[group * 32 + lane];
        Output[size_t(row) * D + slice * (D / SLICES) + j * 32 + lane] = T(value);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
