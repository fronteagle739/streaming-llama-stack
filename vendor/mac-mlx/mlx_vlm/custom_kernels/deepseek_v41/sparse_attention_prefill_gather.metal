// Shared K=V: gather each selected row once, independent of the head count.
// One contiguous BF16 buffer; invalid rows are zero and remain masked by IDs
// in the attention kernel. No threadgroup storage or synchronization needed.
const uint dim = thread_position_in_grid.x % D;
const uint slot = thread_position_in_grid.x / D;
const size_t query = thread_position_in_grid.y;
if (slot >= uint(K)) return;
const uint batch = uint(query / M);
const int id = Indices[query * K + slot];
KV_T value = KV_T(0.0f);
if (id >= 0 && id < NW + NC)
    value = id < NW ? Window[(size_t(batch) * NW + id) * D + dim]
        : Compressed[(size_t(batch) * NC + id - NW) * D + dim];
Gathered[(query * K + slot) * D + dim] = value;
