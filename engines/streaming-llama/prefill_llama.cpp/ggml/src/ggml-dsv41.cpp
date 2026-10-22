#include "ggml.h"
#include "ggml-impl.h"
#include <cmath>

// Native V4.1 fused boundaries. Existing V4 operators retain their IDs.
struct ggml_tensor * ggml_dsv41_engram_add(
        struct ggml_context * ctx, struct ggml_tensor * residual,
        struct ggml_tensor * kv, struct ggml_tensor * q_weight,
        struct ggml_tensor * k_weight, struct ggml_tensor * text_mask, float eps) {
    GGML_ASSERT(residual->type == GGML_TYPE_F32 && ggml_is_contiguous(residual));
    GGML_ASSERT(kv->type == GGML_TYPE_F32 && ggml_is_contiguous(kv));
    GGML_ASSERT(q_weight->type == GGML_TYPE_F32 && ggml_is_contiguous(q_weight));
    GGML_ASSERT(k_weight->type == GGML_TYPE_F32 && ggml_is_contiguous(k_weight));
    GGML_ASSERT(residual->ne[0] > 0 && residual->ne[1] == 4 && residual->ne[3] == 1);
    GGML_ASSERT(residual->ne[2] > 0 && residual->ne[2] <= INT32_MAX / 4);
    GGML_ASSERT(kv->ne[0] == residual->ne[0] && kv->ne[1] == 5 &&
                kv->ne[2] == residual->ne[2] && kv->ne[3] == 1);
    GGML_ASSERT(q_weight->ne[0] == residual->ne[0] && q_weight->ne[1] == 4 &&
                q_weight->ne[2] == 1 && q_weight->ne[3] == 1);
    GGML_ASSERT(ggml_are_same_shape(q_weight, k_weight));
    GGML_ASSERT(eps > 0.f && std::isfinite(eps));
    if (text_mask) {
        GGML_ASSERT(text_mask->type == GGML_TYPE_I8 && ggml_is_contiguous(text_mask));
        GGML_ASSERT(ggml_nelements(text_mask) == residual->ne[2]);
    }
    struct ggml_tensor * result = ggml_dup_tensor(ctx, residual);
    result->op = GGML_OP_DSV41_ENGRAM_ADD;
    result->src[0] = residual;
    result->src[1] = kv;
    result->src[2] = q_weight;
    result->src[3] = k_weight;
    result->src[4] = text_mask;
    ggml_set_op_params(result, &eps, sizeof(eps));
    return result;
}

struct ggml_tensor * ggml_dsv41_pool2(
        struct ggml_context * ctx, struct ggml_tensor * kv,
        struct ggml_tensor * scores, struct ggml_tensor * previous_kv,
        struct ggml_tensor * previous_scores) {
    GGML_ASSERT(kv->type == GGML_TYPE_F32 && ggml_is_contiguous(kv));
    GGML_ASSERT(scores->type == GGML_TYPE_F32 && ggml_is_contiguous(scores));
    GGML_ASSERT(ggml_are_same_shape(kv, scores) && kv->ne[2] == 1 && kv->ne[3] == 1);
    GGML_ASSERT(kv->ne[0] > 0 && kv->ne[1] > 0);
    GGML_ASSERT((previous_kv != NULL) == (previous_scores != NULL));
    if (previous_kv) {
        GGML_ASSERT(previous_kv->type == GGML_TYPE_F32 && ggml_is_contiguous(previous_kv));
        GGML_ASSERT(previous_scores->type == GGML_TYPE_F32 && ggml_is_contiguous(previous_scores));
        GGML_ASSERT(ggml_nelements(previous_kv) == kv->ne[0]);
        GGML_ASSERT(ggml_are_same_shape(previous_kv, previous_scores));
    }
    const int64_t pairs = (kv->ne[1] + (previous_kv != NULL)) / 2;
    GGML_ASSERT(pairs > 0);
    struct ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv->ne[0], pairs);
    result->op = GGML_OP_DSV41_POOL2;
    result->src[0] = kv;
    result->src[1] = scores;
    result->src[2] = previous_kv;
    result->src[3] = previous_scores;
    return result;
}

struct ggml_tensor * ggml_dsv41_rope(
        struct ggml_context * ctx, struct ggml_tensor * x,
        struct ggml_tensor * positions, struct ggml_tensor * frequencies, bool inverse) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && ggml_is_contiguous(x));
    GGML_ASSERT(x->ne[0] >= 64 && x->ne[1] > 0 && x->ne[2] > 0 && x->ne[3] == 1);
    GGML_ASSERT(x->ne[1] <= INT32_MAX / x->ne[2]);
    GGML_ASSERT(positions->type == GGML_TYPE_I32 && ggml_is_contiguous(positions));
    GGML_ASSERT(ggml_nelements(positions) == x->ne[2]);
    GGML_ASSERT(frequencies->type == GGML_TYPE_F32 && ggml_is_contiguous(frequencies));
    GGML_ASSERT(ggml_nelements(frequencies) == 32);
    struct ggml_tensor * result = ggml_dup_tensor(ctx, x);
    const int32_t flag = inverse ? 1 : 0;
    ggml_set_op_params(result, &flag, sizeof(flag));
    result->op = GGML_OP_DSV41_ROPE;
    result->src[0] = x;
    result->src[1] = positions;
    result->src[2] = frequencies;
    return result;
}

struct ggml_tensor * ggml_dsv41_raw_commit(
        struct ggml_context * ctx, struct ggml_tensor * ring,
        struct ggml_tensor * current, struct ggml_tensor * attention_complete, uint32_t start) {
    GGML_ASSERT(ring->type == GGML_TYPE_F32 && ggml_is_contiguous(ring));
    GGML_ASSERT(ring->ne[0] == 512 && ring->ne[1] == 128 && ring->ne[2] == 1 && ring->ne[3] == 1);
    GGML_ASSERT(current->type == GGML_TYPE_F32 && ggml_is_contiguous(current));
    GGML_ASSERT(current->ne[0] == 512 && current->ne[1] > 0 && current->ne[1] <= 32 && current->ne[2] == 1 && current->ne[3] == 1);
    GGML_ASSERT(start < 1048576 && uint64_t(start) + current->ne[1] <= 1048576);
    GGML_ASSERT(attention_complete && attention_complete->type == GGML_TYPE_F32);
    // An explicit data dependency makes old-ring consumers finish before its
    // aliased write. Do not replace this with a zero-valued arithmetic trick:
    // NaNs/infinities in that path can change the KV bits being published.
    struct ggml_tensor * result = ggml_view_tensor(ctx, ring);
    // Persistent cache storage lives in a separate context. Layer-major
    // execution can bypass a full allocation pass, so bind the external view
    // just as the existing V4 persistent-state helpers do.
    result->data = ring->data;
    result->buffer = ring->buffer;
    result->op = GGML_OP_DSV41_RAW_COMMIT;
    result->src[0] = current;
    result->src[1] = ring;
    result->src[2] = attention_complete;
    ggml_set_op_params(result, &start, sizeof(start));
    return result;
}

static void dsv41_visible(const ggml_tensor * visible, int64_t queries) {
    GGML_ASSERT(visible->type == GGML_TYPE_I32 && ggml_is_contiguous(visible) && ggml_nelements(visible) == queries);
    GGML_ASSERT(queries > 0 && queries <= 4096);
}

struct ggml_tensor * ggml_dsv41_index_score(
        struct ggml_context * ctx, struct ggml_tensor * q, struct ggml_tensor * weights,
        struct ggml_tensor * keys, struct ggml_tensor * visible, struct ggml_tensor * block_mask) {
    GGML_ASSERT(q->type == GGML_TYPE_F32 && ggml_is_contiguous(q) && q->ne[0] == 128 && q->ne[1] == 32 && q->ne[3] == 1);
    GGML_ASSERT(weights->type == GGML_TYPE_F32 && ggml_is_contiguous(weights) && weights->ne[0] == 32 && weights->ne[1] == q->ne[2] && weights->ne[2] == 1 && weights->ne[3] == 1);
    GGML_ASSERT(keys->type == GGML_TYPE_F32 && ggml_is_contiguous(keys) && keys->ne[0] == 128 && keys->ne[1] > 0 && keys->ne[1] <= 1048576 && keys->ne[2] == 1 && keys->ne[3] == 1);
    dsv41_visible(visible,q->ne[2]);
    if(block_mask) GGML_ASSERT(block_mask->type == GGML_TYPE_F32 && ggml_is_contiguous(block_mask) && block_mask->ne[0] == (keys->ne[1]+7)/8 && block_mask->ne[1] == q->ne[2] && block_mask->ne[2] == 1 && block_mask->ne[3] == 1);
    auto * result = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,keys->ne[1],q->ne[2]);
    result->op=GGML_OP_DSV41_INDEX_SCORE; result->src[0]=q;result->src[1]=weights;result->src[2]=keys;result->src[3]=visible;result->src[4]=block_mask;
    return result;
}

struct ggml_tensor * ggml_dsv41_candidate_blocks(
        struct ggml_context * ctx, struct ggml_tensor * scores, struct ggml_tensor * visible) {
    GGML_ASSERT(scores->type==GGML_TYPE_F32 && ggml_is_contiguous(scores) && scores->ne[0]>0 && scores->ne[0]<=1048576 && scores->ne[2]==1 && scores->ne[3]==1);
    dsv41_visible(visible,scores->ne[1]);
    auto * result=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,(scores->ne[0]+7)/8,scores->ne[1]);
    result->op=GGML_OP_DSV41_CANDIDATE_BLOCKS;result->src[0]=scores;result->src[1]=visible;
    return result;
}

struct ggml_tensor * ggml_dsv41_selected_mask(
        struct ggml_context * ctx, struct ggml_tensor * selected, struct ggml_tensor * visible, struct ggml_tensor * block_mask) {
    GGML_ASSERT(selected->type==GGML_TYPE_I32 && ggml_is_contiguous(selected) && selected->ne[0]>0 && selected->ne[0]<=512 && selected->ne[2]==1 && selected->ne[3]==1);
    dsv41_visible(visible,selected->ne[1]);
    if(block_mask) GGML_ASSERT(block_mask->type==GGML_TYPE_F32 && ggml_is_contiguous(block_mask) && block_mask->ne[0]>0 && block_mask->ne[0]<=131072 && block_mask->ne[1]==selected->ne[1] && block_mask->ne[2]==1 && block_mask->ne[3]==1);
    auto * result=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,selected->ne[0],selected->ne[1]);
    result->op=GGML_OP_DSV41_SELECTED_MASK;result->src[0]=selected;result->src[1]=visible;result->src[2]=block_mask;
    return result;
}

struct ggml_tensor * ggml_dsv41_masked_half(
        struct ggml_context * ctx, struct ggml_tensor * gathered, struct ggml_tensor * mask) {
    GGML_ASSERT(gathered->type == GGML_TYPE_F32 && ggml_is_contiguous(gathered));
    GGML_ASSERT(gathered->ne[0] > 0 && gathered->ne[0] <= 512 && gathered->ne[1] > 0 && gathered->ne[1] <= 640);
    GGML_ASSERT(gathered->ne[2] > 0 && gathered->ne[2] <= 4096 && gathered->ne[3] == 1);
    GGML_ASSERT(mask->type == GGML_TYPE_F32 && ggml_is_contiguous(mask));
    GGML_ASSERT(mask->ne[0] == gathered->ne[1] && mask->ne[1] == gathered->ne[2] && mask->ne[2] == 1 && mask->ne[3] == 1);
    auto * result = ggml_dup_tensor(ctx, gathered);
    result->op = GGML_OP_DSV41_MASKED_HALF; result->src[0] = gathered; result->src[1] = mask;
    return result;
}
