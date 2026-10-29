#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include "../ggml-dsv41-ops.h"

// One native 32-lane reduction for each of the four residual copies.
static __global__ void dsv41_engram_add_kernel(
        float * out, const float * residual, const float * kv,
        const float * qw, const float * kw, const int8_t * mask,
        int64_t width, int64_t rows, float eps) {
    const int64_t row = blockIdx.x;
    if (row >= rows) return;
    const int lane = threadIdx.x;
    const int64_t token = row / 4, copy = row % 4;
    const float * h = residual + row * width;
    float * result = out + row * width;
    if (mask && !mask[token]) {
        for (int64_t i = lane; i < width; i += 32) result[i] = h[i];
        return;
    }
    const float * key = kv + (token * 5 + copy) * width;
    const float * value = kv + (token * 5 + 4) * width;
    float h2 = 0.f, k2 = 0.f, dot = 0.f;
    for (int64_t i = lane; i < width; i += 32) {
        const float k = ggml_dsv41_bf16(key[i]);
        h2 += h[i] * h[i];
        k2 += k * k;
        dot += h[i] * (qw[copy * width + i] * kw[copy * width + i]) * k;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        h2 += __shfl_down_sync(0xffffffffu, h2, offset);
        k2 += __shfl_down_sync(0xffffffffu, k2, offset);
        dot += __shfl_down_sync(0xffffffffu, dot, offset);
    }
    float gate = 0.f;
    if (lane == 0) {
        dot *= rsqrtf(float(h2 / width) + eps);
        dot *= rsqrtf(float(k2 / width) + eps);
        dot *= rsqrtf(float(width));
        gate = ggml_dsv41_engram_gate(dot);
    }
    gate = __shfl_sync(0xffffffffu, gate, 0);
    for (int64_t i = lane; i < width; i += 32) {
        result[i] = ggml_dsv41_bf16(h[i] + gate * ggml_dsv41_bf16(value[i]));
    }
}

static __global__ void dsv41_pool2_kernel(
        float * out, const float * kv, const float * scores,
        const float * previous_kv, const float * previous_scores,
        int64_t width, int64_t count) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int64_t col = i % width, pair = i / width;
    const int64_t a = pair * 2 - (previous_kv != nullptr), b = a + 1;
    const float ka = a < 0 ? previous_kv[col] : kv[a * width + col];
    const float sa = a < 0 ? previous_scores[col] : scores[a * width + col];
    out[i] = ggml_dsv41_pool2_value(ka, kv[b * width + col], sa, scores[b * width + col]);
}

static __global__ void dsv41_rope_kernel(
        float * out, const float * input, const int32_t * positions,
        const float * frequencies, int64_t width, int64_t heads, int64_t rows, bool inverse, bool round_input) {
    const int64_t row = blockIdx.x;
    if (row >= rows) return;
    const int lane = threadIdx.x;
    // round_input folds the preceding exact BF16 round trip into this pass:
    // every element is read once, rounded with the same ggml_dsv41_bf16 as
    // the QAT_BF16 operator, and only then copied or rotated.
    if (round_input)
        for (int64_t i = lane; i < width - 64; i += 32) out[row * width + i] = ggml_dsv41_bf16(input[row * width + i]);
    else if (out != input)
        for (int64_t i = lane; i < width - 64; i += 32) out[row * width + i] = input[row * width + i];
    const int64_t offset = row * width + width - 64 + 2 * lane;
    const float theta = ggml_dsv41_rope_product(float(positions[row / heads]), frequencies[lane]);
    const float re = round_input ? ggml_dsv41_bf16(input[offset]) : input[offset];
    const float im = round_input ? ggml_dsv41_bf16(input[offset + 1]) : input[offset + 1];
    const auto result = ggml_dsv41_rope_pair(re, im, theta, inverse);
    out[offset] = result.re; out[offset + 1] = result.im;
}

static __global__ void dsv41_raw_commit_kernel(float * ring, const float * current, uint32_t start, int64_t count) {
    const int64_t row = blockIdx.x;
    if (row >= count) return;
    for (int i = threadIdx.x; i < 512; i += blockDim.x)
        ring[((start + row) % 128) * 512 + i] = current[row * 512 + i];
}

// Four key rows per block, one warp per row. Reduce 128 dimensions and 32
// heads in registers; never materialize [context,heads,queries] intermediates.
static __global__ void dsv41_index_score_kernel(float * out,const float * q,const float * weights,
        const float * keys,const int32_t * visible,const float * mask,int64_t width) {
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int64_t row=int64_t(blockIdx.x)*4+warp,token=blockIdx.y,blocks=(width+7)/8;
    if(row>=width)return;
    if(row>=visible[token] || (mask && mask[token*blocks+row/8]!=0.f)){
        if(!lane)out[token*width+row]=-INFINITY;return;
    }
    float sum=0;
    for(int h=0;h<32;++h){
        float dot=0;
        for(int d=lane;d<128;d+=32)dot=ggml_dsv41_rope_fma(q[(token*32+h)*128+d],keys[row*128+d],dot);
        for(int offset=16;offset;offset/=2)dot=ggml_dsv41_add_rn(dot,__shfl_down_sync(0xffffffffu,dot,offset));
        if(!lane)sum=ggml_dsv41_rope_fma(ggml_dsv41_relu(ggml_dsv41_rope_product(dot,1.f/64.f)),weights[token*32+h],sum);
    }
    if(!lane)out[token*width+row]=sum;
}

static __global__ void dsv41_candidate_blocks_kernel(float * out,const float * scores,const int32_t * visible,int64_t width,int64_t queries,int64_t key_offset){
    const int64_t blocks=(width+7)/8,i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=blocks*queries)return;
    const int64_t token=i/blocks,block=i%blocks;
    const int64_t local_visible=min(width,max(int64_t(0),int64_t(visible[token])-key_offset));
    float best=-INFINITY;
    for(int64_t row=block*8;row<local_visible && row<(block+1)*8;++row)best=fmaxf(best,scores[token*width+row]);
    out[i]=visible[token]>key_offset && key_offset/8+block==(visible[token]-1)/8 ? INFINITY : best;
}

static __global__ void dsv41_selected_mask_kernel(float * out,const int32_t * ids,const int32_t * visible,const float * mask,
        int64_t blocks,int64_t top,int64_t count){
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
    const int64_t token=i/top,id=ids[i];
    out[i]=id>=0 && id<visible[token] && (!mask || (id/8<blocks && mask[token*blocks+id/8]==0.f)) ? 0.f : -INFINITY;
}

static __global__ void dsv41_masked_half_kernel(float * out, const float * input, const float * mask,
        int64_t width, int64_t count) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (__float_as_uint(mask[i / width]) == 0xff800000u) { out[i] = 0.f; return; }
    out[i] = __half2float(__float2half_rn(input[i]));
}
