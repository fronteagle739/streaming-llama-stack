// SPDX-License-Identifier: MIT
// Ported from https://github.com/IngeniousIdiocy/ds4-v41-m3ultra at c8aa547,
// metal/argsort.metal:52-147. See V41_KERNELS_NOTICE.md.

constexpr bool causal = false, shuffle = false, prefix = false;
constexpr int DS4_SORT_ORDER_ASC = 0, order = 1;
const uint3 tgpig = threadgroup_position_in_grid;
const ushort3 tpitg = ushort3(thread_position_in_threadgroup);
const ushort3 ntg = ushort3(THREADS, 1, 1);
device const char *src0 = (device const char *)Scores;
device int32_t *dst = Output;
threadgroup int32_t shmem_i32[THREADS*2];

    // bitonic sort
    const int col = tpitg[0];
    const int ib  = tgpig[0] / ROWS;

    const int i00 = ib*ntg.x;
    const int i01 = tgpig[0] % ROWS;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];
    const int width = causal ? min(N,
        int((0u + uint(i01) + 1u) / 1u)) : N;
    if (i00 >= width) return;
    const int work_width = causal ?
        ((width - 1) / ntg.x) * TOPK + min((width - 1) % ntg.x + 1, TOPK) : WORK;

    device const float * src0_row = (device const float *) (src0 + (size_t(N)*sizeof(float))*i01 + (size_t(N)*ROWS*sizeof(float))*i02 + (size_t(N)*ROWS*sizeof(float))*i03);

    // initialize indices
    shmem_i32[col] = i00 + col;

    // Stage this block's score slice in threadgroup memory (indices stay in
    // [i00, i00+ntg.x), so shmem_f32[idx - i00] replaces the device gather).
    // The host allocates ntg.x extra floats after the index array.  Values and
    // the comparison network are unchanged, so the permutation is identical.
    threadgroup float * shmem_f32 = (threadgroup float *) (shmem_i32 + ntg.x);
    if (i00 + col < width) {
        shmem_f32[col] = src0_row[i00 + col];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    int reg_idx = i00 + col;
    float reg_value = reg_idx < width ? shmem_f32[col] : 0.0f;
    for (int k = 2; k <= ntg.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            if (shuffle && j < 32) {
                if (k > 32 && j == 16) {
                    reg_idx = shmem_i32[col];
                    reg_value = reg_idx < width ? shmem_f32[reg_idx - i00] : 0.0f;
                }
                const int other = simd_shuffle_xor(reg_idx, j);
                const float value = simd_shuffle_xor(reg_value, j);
                const bool first = ((col & k) == 0) == ((col & j) == 0);
                const bool exchange = first ?
                    (reg_idx >= width || (other < width && (order == DS4_SORT_ORDER_ASC ?
                        reg_value > value : reg_value < value))) :
                    (other >= width || (reg_idx < width && (order == DS4_SORT_ORDER_ASC ?
                        reg_value < value : reg_value > value)));
                if (exchange) { reg_idx = other; reg_value = value; }
                continue;
            }
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (shmem_i32[col] >= width ||
                       (shmem_i32[ixj] <  width && (order == DS4_SORT_ORDER_ASC ?
                            shmem_f32[shmem_i32[col] - i00] > shmem_f32[shmem_i32[ixj] - i00] :
                            shmem_f32[shmem_i32[col] - i00] < shmem_f32[shmem_i32[ixj] - i00]))
                    ) {
                        int swap_tmp = shmem_i32[col];
                        shmem_i32[col] = shmem_i32[ixj];
                        shmem_i32[ixj] = swap_tmp;
                    }
                } else {
                    if (shmem_i32[ixj] >= width ||
                       (shmem_i32[col] <  width && (order == DS4_SORT_ORDER_ASC ?
                            shmem_f32[shmem_i32[col] - i00] < shmem_f32[shmem_i32[ixj] - i00] :
                            shmem_f32[shmem_i32[col] - i00] > shmem_f32[shmem_i32[ixj] - i00]))
                    ) {
                        int swap_tmp = shmem_i32[col];
                        shmem_i32[col] = shmem_i32[ixj];
                        shmem_i32[ixj] = swap_tmp;
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (shuffle && k >= 32) {
            shmem_i32[col] = reg_idx;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int64_t i0 = ib*TOPK;

    // copy the result to dst without the padding
    if (i0 + col < work_width && col < TOPK) {
        dst += i0 + WORK*i01 + WORK*ROWS*i02 + WORK*ROWS*1*i03;

        dst[col] = shuffle ? reg_idx : shmem_i32[col];
    }
