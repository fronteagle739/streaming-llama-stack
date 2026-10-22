#include "ops.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "ggml-dsv41-ops.h"
#include "ggml-dsv41-index-window.h"
#include "ggml-dsv41-gather-half.h"
#include <algorithm>

void ggml_compute_forward_dsv41_engram_add(
        const ggml_compute_params * params, ggml_tensor * dst) {
    const int64_t width = dst->ne[0];
    const int64_t rows = 4 * dst->ne[2];
    const float * residual = (const float *) dst->src[0]->data;
    const float * kv = (const float *) dst->src[1]->data;
    const float * qw = (const float *) dst->src[2]->data;
    const float * kw = (const float *) dst->src[3]->data;
    const int8_t * mask = dst->src[4] ? (const int8_t *) dst->src[4]->data : nullptr;
    float * out = (float *) dst->data;
    const float eps = ggml_get_op_params_f32(dst, 0);
    for (int64_t row = params->ith; row < rows; row += params->nth) {
        const int64_t token = row / 4, copy = row % 4;
        const float * h = residual + row * width;
        float * result = out + row * width;
        if (mask && !mask[token]) {
            memcpy(result, h, width * sizeof(float));
            continue;
        }
        const float * key = kv + (token * 5 + copy) * width;
        const float * value = kv + (token * 5 + 4) * width;
        float h2 = 0.f, k2 = 0.f, dot = 0.f;
        for (int64_t i = 0; i < width; ++i) {
            const float k = ggml_dsv41_bf16(key[i]);
            h2 += h[i] * h[i];
            k2 += k * k;
            dot += h[i] * (qw[copy * width + i] * kw[copy * width + i]) * k;
        }
        dot *= 1.f / sqrtf(h2 / width + eps);
        dot *= 1.f / sqrtf(k2 / width + eps);
        dot *= 1.f / sqrtf(float(width));
        const float gate = ggml_dsv41_engram_gate(dot);
        for (int64_t i = 0; i < width; ++i) {
            result[i] = ggml_dsv41_bf16(h[i] + gate * ggml_dsv41_bf16(value[i]));
        }
    }
}

void ggml_compute_forward_dsv41_pool2(
        const ggml_compute_params * params, ggml_tensor * dst) {
    const int64_t width = dst->ne[0], count = ggml_nelements(dst);
    const float * kv = (const float *) dst->src[0]->data;
    const float * scores = (const float *) dst->src[1]->data;
    const float * previous_kv = dst->src[2] ? (const float *) dst->src[2]->data : nullptr;
    const float * previous_scores = dst->src[3] ? (const float *) dst->src[3]->data : nullptr;
    float * out = (float *) dst->data;
    for (int64_t i = params->ith; i < count; i += params->nth) {
        const int64_t col = i % width, pair = i / width;
        const int64_t a = pair * 2 - (previous_kv != nullptr), b = a + 1;
        const float ka = a < 0 ? previous_kv[col] : kv[a * width + col];
        const float sa = a < 0 ? previous_scores[col] : scores[a * width + col];
        out[i] = ggml_dsv41_pool2_value(ka, kv[b * width + col], sa, scores[b * width + col]);
    }
}

void ggml_compute_forward_dsv41_rope(const ggml_compute_params * params, ggml_tensor * dst) {
    const int64_t width = dst->ne[0], heads = dst->ne[1], rows = heads * dst->ne[2];
    const float * input = (const float *) dst->src[0]->data;
    const int32_t * positions = (const int32_t *) dst->src[1]->data;
    const float * frequencies = (const float *) dst->src[2]->data;
    float * output = (float *) dst->data;
    const bool inverse = ggml_get_op_params_i32(dst, 0) != 0;
    for (int64_t row = params->ith; row < rows; row += params->nth) {
        const int32_t pos = positions[row / heads];
        GGML_ASSERT(pos >= 0 && pos < 1048576);
        if (output != input) memcpy(output + row * width, input + row * width, (width - 64) * sizeof(float));
        for (int i = 0; i < 32; ++i) {
            const int64_t offset = row * width + width - 64 + 2 * i;
            const float theta = float(pos) * frequencies[i];
            const auto result = ggml_dsv41_rope_pair(input[offset], input[offset + 1], theta, inverse);
            output[offset] = result.re; output[offset + 1] = result.im;
        }
    }
}

void ggml_compute_forward_dsv41_raw_commit(const ggml_compute_params * params, ggml_tensor * dst) {
    const uint32_t start = uint32_t(ggml_get_op_params_i32(dst, 0));
    const float * input = (const float *) dst->src[0]->data;
    float * ring = (float *) dst->data;
    for (int64_t t = params->ith; t < dst->src[0]->ne[1]; t += params->nth)
        memcpy(ring + ((start + t) % 128) * 512, input + t * 512, 512 * sizeof(float));
}

void ggml_compute_forward_dsv41_index_score(const ggml_compute_params * params, ggml_tensor * dst) {
    const int64_t width=dst->ne[0], blocks=(width+7)/8, count=ggml_nelements(dst);
    const float * q=(const float*)dst->src[0]->data,*weights=(const float*)dst->src[1]->data,*keys=(const float*)dst->src[2]->data;
    const int32_t * visible=(const int32_t*)dst->src[3]->data;
    const float * mask=dst->src[4]?(const float*)dst->src[4]->data:nullptr;
    float * out=(float*)dst->data;
    for(int64_t i=params->ith;i<count;i+=params->nth) {
        const int64_t token=i/width,key=i%width;
        GGML_ASSERT(visible[token]>=0 && visible[token]<=width);
        if(key>=visible[token] || (mask && mask[token*blocks+key/8]!=0.f)){out[i]=-INFINITY;continue;}
        float sum=0;
        for(int h=0;h<32;++h){
            float dot=0;for(int d=0;d<128;++d)dot=fmaf(q[(token*32+h)*128+d],keys[key*128+d],dot);
            sum=fmaf(ggml_dsv41_relu(ggml_dsv41_rope_product(dot,1.f/64.f)),weights[token*32+h],sum);
        }
        out[i]=sum;
    }
}

void ggml_compute_forward_dsv41_candidate_blocks(const ggml_compute_params * params, ggml_tensor * dst) {
    const int64_t width=dst->src[0]->ne[0],blocks=dst->ne[0],count=ggml_nelements(dst);
    const float * scores=(const float*)dst->src[0]->data;const int32_t * visible=(const int32_t*)dst->src[1]->data;
    float * out=(float*)dst->data;
    const int64_t key_offset=ggml_dsv41_index_window_offset(dst);
    for(int64_t i=params->ith;i<count;i+=params->nth){
        const int64_t token=i/blocks,block=i%blocks;GGML_ASSERT(visible[token]>=0 && visible[token]<=(ggml_dsv41_index_window_requested(dst)?1048576:width));
        const int64_t local_visible=std::min(width,std::max<int64_t>(0,int64_t(visible[token])-key_offset));
        float best=-INFINITY;
        for(int64_t row=block*8;row<std::min<int64_t>(local_visible,(block+1)*8);++row)best=fmaxf(best,scores[token*width+row]);
        out[i]=visible[token]>key_offset && key_offset/8+block==(visible[token]-1)/8 ? INFINITY : best;
    }
}

void ggml_compute_forward_dsv41_selected_mask(const ggml_compute_params * params, ggml_tensor * dst) {
    const int64_t top=dst->ne[0],count=ggml_nelements(dst),blocks=dst->src[2]?dst->src[2]->ne[0]:0;
    const int32_t * ids=(const int32_t*)dst->src[0]->data,*visible=(const int32_t*)dst->src[1]->data;
    const float * mask=blocks?(const float*)dst->src[2]->data:nullptr;float * out=(float*)dst->data;
    for(int64_t i=params->ith;i<count;i+=params->nth){
        const int64_t token=i/top,id=ids[i];GGML_ASSERT(visible[token]>=0 && visible[token]<=1048576);
        const bool valid=id>=0 && id<visible[token] && (!mask || (id/8<blocks && mask[token*blocks+id/8]==0.f));
        out[i]=valid?0.f:-INFINITY;
    }
}

void ggml_compute_forward_dsv41_masked_half(const ggml_compute_params * params, ggml_tensor * dst) {
    const bool gather=ggml_dsv41_gather_half_requested(dst);
    GGML_ASSERT(gather?ggml_dsv41_gather_half_valid(dst):!dst->src[2]);
    const int64_t width = dst->ne[0], rows = dst->ne[1] * dst->ne[2];
    const float * input = (const float *) dst->src[0]->data, * mask = (const float *) dst->src[1]->data;
    const auto * extra=gather?dst->src[3]:nullptr;
    float * output = (float *) dst->data;
    for (int64_t row = params->ith; row < rows; row += params->nth) {
        const int64_t source_row=gather?((const int32_t *)dst->src[2]->data)[row]:row;
        GGML_ASSERT(!gather || (source_row>=0 && source_row<dst->src[0]->ne[1]+(extra?extra->ne[1]:0)));
        // Bit comparison retains the skip under -ffast-math. Do not read a
        // masked KV element: 0*Inf/NaN is not a conditional masked load.
        uint32_t bits; memcpy(&bits, mask + row, sizeof(bits));
        if (bits == 0xff800000u) {
            memset(output + row * width, 0, width * sizeof(float));
            continue;
        }
        for (int64_t d = 0; d < width; ++d){
            float value;
            if(extra&&source_row>=dst->src[0]->ne[1]){
                const int64_t at=(source_row-dst->src[0]->ne[1])*width+d;
                value=extra->type==GGML_TYPE_BF16?ggml_bf16_to_fp32(((const ggml_bf16_t*)extra->data)[at]):((const float*)extra->data)[at];
            }else value=input[source_row*width+d];
            output[row * width + d] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
        }
    }
}
