// SPDX-License-Identifier: MIT
// Ported from https://github.com/IngeniousIdiocy/ds4-v41-m3ultra at c8aa547,
// metal/argsort.metal:166-315. See V41_KERNELS_NOTICE.md.

constexpr bool causal = false, shuffle = false, prefix = false;
constexpr int DS4_SORT_ORDER_ASC = 0, order = 1;
const uint3 tgpig = threadgroup_position_in_grid;
const ushort3 tpitg = ushort3(thread_position_in_threadgroup);
const ushort3 ntg = ushort3(THREADS, 1, 1);
device const char *src0 = (device const char *)Scores;
device int32_t *dst = Output;
device const int32_t *tmp = Work;


    const int im  = tgpig[0] / ROWS;
    const int i01 = tgpig[0] % ROWS;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    const int start = im * (2 * RUN);
    const uint width = causal ? min(uint(N),
        (0u + uint(i01) + 1u) / 1u) : 0u;
    const int work_width = causal ? int(((width - 1u) / THREADS) * BLOCK_TOPK +
        min((width - 1u) % THREADS + 1u, uint(BLOCK_TOPK))) : WORK;

    // A merged run can contribute at most 512 entries to the final result.
    // Keep the original run offsets so the comparison and tie order agree.
    const int read_limit = prefix ? min(RUN, 512) : RUN;
    const int len0 = min(read_limit, max(0, work_width - start));
    const int len1 = min(read_limit, max(0, work_width - (start + RUN)));

    const int total = len0 + len1;

    device const int32_t * tmp0 = tmp + start
        + i01*WORK
        + i02*WORK*ROWS
        + i03*WORK*ROWS*1;

    device const int32_t * tmp1 = tmp0 + RUN;

    dst += start
        + i01*TOPK
        + i02*TOPK*ROWS
        + i03*TOPK*ROWS*1;

    device const float * src0_row = (device const float *)(src0
        + (size_t(N)*sizeof(float))*i01
        + (size_t(N)*ROWS*sizeof(float))*i02
        + (size_t(N)*ROWS*sizeof(float))*i03);

    if (total == 0) {
        return;
    }

    const int chunk = (total + ntg.x - 1) / ntg.x;

    const int k0 = tpitg.x * chunk;
    const int limit = prefix ? min(TOPK, 512) : TOPK;
    const int k1 = min(min(k0 + chunk, total), limit);

    if (k0 >= limit) {
        return;
    }

    if (k0 >= total) {
        return;
    }

    int low  = k0 > len1 ? k0 - len1 : 0;
    int high = min(k0, len0);

    // binary-search partition (i, j) such that i + j = k
    while (low < high) {
        const int mid = (low + high) >> 1;

        const int32_t idx0 = tmp0[mid];
        const int32_t idx1 = tmp1[k0 - mid - 1];

        const float val0 = src0_row[idx0];
        const float val1 = src0_row[idx1];

        bool take_left;
        if (order == DS4_SORT_ORDER_ASC) {
            take_left = (val0 <= val1);
        } else {
            take_left = (val0 >= val1);
        }

        if (take_left) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    int i = low;
    int j = k0 - i;

    // keep the merge fronts into registers
    int32_t idx0 = 0;
    float   val0 = 0.0f;
    if (i < len0) {
        idx0 = tmp0[i];
        val0 = src0_row[idx0];
    }

    int32_t idx1 = 0;
    float   val1 = 0.0f;
    if (j < len1) {
        idx1 = tmp1[j];
        val1 = src0_row[idx1];
    }

    for (int k = k0; k < k1; ++k) {
        int32_t out_idx;

        if (i >= len0) {
            while (k < k1) {
                dst[k++] = tmp1[j++];
            }
            break;
        } else if (j >= len1) {
            while (k < k1) {
                dst[k++] = tmp0[i++];
            }
            break;
        } else {
            bool take_left;

            if (order == DS4_SORT_ORDER_ASC) {
                take_left = (val0 <= val1);
            } else {
                take_left = (val0 >= val1);
            }

            if (take_left) {
                out_idx = idx0;
                ++i;
                if (i < len0) {
                    idx0 = tmp0[i];
                    val0 = src0_row[idx0];
                }
            } else {
                out_idx = idx1;
                ++j;
                if (j < len1) {
                    idx1 = tmp1[j];
                    val1 = src0_row[idx1];
                }
            }
        }

        dst[k] = out_idx;
    }
