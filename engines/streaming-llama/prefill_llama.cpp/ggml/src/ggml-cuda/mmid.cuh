#pragma once

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, cudaStream_t stream);

// Invert a permutation: given perm[i] = j, produce inv[j] = i.
// Used to convert mm_ids_helper's scatter mapping (ids_dst) into the gather
// mapping needed by get_rows_cuda for the fallback mul_mat_id path.
void ggml_cuda_invert_permutation(const int32_t * perm, int32_t * inv, int n, cudaStream_t stream);
