const uint tid = thread_index_in_threadgroup;
const uint lane = tid % 32;
const uint sg = tid / 32;
const uint row = threadgroup_position_in_grid.y;
const uint query = row / H;
threadgroup float scratch[8];
float maximum = max(float(Sink[row % H]), -1e30f);
for (uint slot = tid; slot < K; slot += 256)
    maximum = max(maximum, Scores[size_t(row) * K + slot]);
maximum = simd_max(maximum);
if (lane == 0) scratch[sg] = maximum;
threadgroup_barrier(mem_flags::mem_threadgroup);
maximum = simd_max(lane < 8 ? scratch[lane] : -1e30f);
float total = 0.0f;
for (uint slot = tid; slot < K; slot += 256) {
    const int index = Indices[size_t(query) * K + slot];
    const float value = index >= 0 && index < NW + NC
        ? exp(Scores[size_t(row) * K + slot] - maximum) : 0.0f;
    Weights[size_t(row) * K + slot] = value;
    total += value;
}
total = simd_sum(total);
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) scratch[sg] = total;
threadgroup_barrier(mem_flags::mem_threadgroup);
total = simd_sum(lane < 8 ? scratch[lane] : 0.0f);
if (tid == 0) Denom[row] = total + exp(float(Sink[row % H]) - maximum);
