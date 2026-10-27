// MLX v0.32.2 Steel/decoder header is included verbatim from expert_tiles_qmm.metal.
// The upstream MIT notice and pinned source manifest are retained there.
// Only scheduling, shared staging and the activation epilogue are new here.
// --- header ---
// Verbatim Sigmoid from MLX v0.32.2 unary_ops.h:294-300, renamed to avoid collisions.
struct FusedFFNSigmoid {
  template <typename T>
  T operator()(T x) thread {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};
// Floating Minimum/Maximum bodies from the same tag's binary_ops.h.
// Preserve NaNs and operand order, including clip's maximum then minimum.
inline float fused_min(float x, float y) {
  if (metal::isnan(x)) {
    return x;
  }
  return x < y ? x : y;
}
inline float fused_max(float x, float y) {
  if (metal::isnan(x)) {
    return x;
  }
  return x > y ? x : y;
}
// --- body ---
constexpr int BM = 32, BN = 32, BK = 32, LD = BK + 16 / sizeof(T);
const uint3 tid = threadgroup_position_in_grid;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int K = X_shape[2], N = GateW_shape[1];
const uint q = tid.y;
if (q >= Prefix[384]) return;  // Uniform cull, before any barriers.
uint lo = 0, hi = 385;
while (lo < hi) {
  const uint mid = lo + (hi - lo) / 2;
  if (Prefix[mid] <= q) lo = mid + 1;
  else hi = mid;
}
const uint expert = lo - 1;
const uint row = Offsets[expert] + BM * (q - Prefix[expert]);
const short m = short(min(uint(BM), Offsets[expert + 1] - row));
const int col = tid.x * BN;
const int row_half = 16 * (sg / 2);
const bool active = row_half < m;

// Two stock BM16/WM1/WN2 blocks, with identical eight-wide K accumulation.
// sg 0/1 own rows 0..15; sg 2/3 own 16..31. Each owns both projections.
// Using WM2 directly would interleave eight-row stripes, breaking half culling.
using mma_t = mlx::steel::BlockMMA<T, float, 16, BN, BK, 1, 2,
                                  false, true, LD, LD>;
using x_loader_t = mlx::steel::BlockLoader<T, BM, BK, LD, 1, 128>;
using w_loader_t = QuantizedBlockLoader<T, BN, BK, LD, 1, 128, 32, 4>;
threadgroup T Xs[BM * LD];
threadgroup T Gs[BN * LD];
threadgroup T Us[BN * LD];
thread mma_t gate(sg % 2, lane), up(sg % 2, lane);
thread x_loader_t load_x(X + size_t(row) * K, K, Xs, sg, lane);
const size_t w_offset = (size_t(expert) * N + col) * (K / 2);
const size_t s_offset = (size_t(expert) * N + col) * (K / 32);
thread w_loader_t load_g((const device uint8_t*)GateW + w_offset,
                        GateS + s_offset, K, Gs, sg, lane);
thread w_loader_t load_u((const device uint8_t*)UpW + w_offset,
                        UpS + s_offset, K, Us, sg, lane);
for (int k = 0; k < K / BK; ++k) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (m == BM) load_x.load_unsafe();
  else load_x.load_safe(short2(BK, m));
  load_g.load_unsafe();
  load_u.load_unsafe();
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // All 128 threads stage and synchronize, even when the upper half is absent.
  if (active) {
    gate.mma(Xs + row_half * LD, Gs);
    up.mma(Xs + row_half * LD, Us);
  }
  load_x.next();
  load_g.next();
  load_u.next();
}
if (active) {
  const float limit = Limit[0];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < decltype(gate.Ctile)::kElemsPerTile; ++i) {
    // Stock gather_qmm stores BF16; ClampedSwiGLU casts those values to FP32.
    // Reproduce BOTH conversions before any clamp or sigmoid operation.
    float g = float(T(gate.Ctile.elems()[i]));
    float u = float(T(up.Ctile.elems()[i]));
    if (limit > 0) {
      g = fused_min(g, limit);
      u = fused_min(fused_max(u, -limit), limit);
    }
    // nn.silu(g) is g * sigmoid(g), NOT g / (1 + exp(-g)).
    const float silu = g * FusedFFNSigmoid{}(g);
    gate.Ctile.elems()[i] = silu * u;
  }
  device float* output = Hidden + size_t(row + row_half) * N + col;
  const short live = min(short(16), short(m - row_half));
  if (live == 16) gate.store_result(output, N);
  else gate.store_result_slice(output, N, short2(0, 0), short2(BN, live));
}
