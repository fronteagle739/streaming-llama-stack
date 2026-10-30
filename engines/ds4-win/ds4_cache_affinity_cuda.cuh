/* Adapt the native router warp top-k reduction to corrected affinity scores.
 * The ascending expert-ID tie rule and sequential unbiased combine are retained. */
#ifndef DS4_CACHE_AFFINITY_CUDA_CUH
#define DS4_CACHE_AFFINITY_CUDA_CUH
#include "ds4_cache_affinity.h"
__device__ __forceinline__ static bool router_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}
template<uint32_t EXPERTS>
__device__ __forceinline__ void ds4_cache_affinity_select_warp_width(
        int32_t *selected, float *weights, const float *probs,
        const float *bias, const uint8_t *resident, float lambda,
        uint32_t used, float scale, float ram_cost) {
    const uint32_t lane=threadIdx.x;
    float local_prob[EXPERTS/32],local_score[EXPERTS/32];
    bool valid[EXPERTS/32];
    #pragma unroll
    for(uint32_t j=0;j<EXPERTS/32;j++) {
        const uint32_t e=lane+j*32u;
        local_prob[j]=probs[e];
        local_score[j]=probs[e]+(bias?bias[e]:0.0f)-ds4_affinity_tier_penalty(resident[e],lambda,ram_cost);
        valid[j]=true;
    }
    float out_prob[6];uint32_t out_idx[6];
    #pragma unroll
    for(uint32_t k=0;k<6u;k++) {
        float best_score=-INFINITY,best_prob=0.0f;uint32_t best_idx=UINT32_MAX;
        #pragma unroll
        for(uint32_t j=0;j<EXPERTS/32;j++) {
            const uint32_t e=lane+j*32u;
            if(valid[j] && router_score_better(local_score[j],e,best_score,best_idx)) {
                best_score=local_score[j];best_prob=local_prob[j];best_idx=e;
            }
        }
        #pragma unroll
        for(uint32_t mask=16u;mask>0u;mask>>=1u) {
            const float other_score=__shfl_xor_sync(0xffffffffu,best_score,mask);
            const float other_prob=__shfl_xor_sync(0xffffffffu,best_prob,mask);
            const uint32_t other_idx=__shfl_xor_sync(0xffffffffu,best_idx,mask);
            if(router_score_better(other_score,other_idx,best_score,best_idx)) {
                best_score=other_score;best_prob=other_prob;best_idx=other_idx;
            }
        }
        #pragma unroll
        for(uint32_t j=0;j<EXPERTS/32;j++)if(lane+j*32u==best_idx)valid[j]=false;
        if(lane==0){out_idx[k]=best_idx;out_prob[k]=best_prob;}
    }
    if(lane==0) {
        float sum=0.0f;
        #pragma unroll
        for(uint32_t j=0;j<6u;j++){selected[j]=(int32_t)out_idx[j];weights[j]=out_prob[j];sum+=out_prob[j];}
        sum=fmaxf(sum,6.103515625e-5f);
        #pragma unroll
        for(uint32_t j=0;j<6u;j++)weights[j]=weights[j]/sum*scale;
    }
    __syncwarp();
}
__device__ __forceinline__ void ds4_cache_affinity_select_warp(
        int32_t *selected, float *weights, const float *probs,
        const float *bias, const uint8_t *resident, float lambda,
        uint32_t experts, uint32_t used, float scale, float ram_cost = 0.0f) {
    if(experts==384u)ds4_cache_affinity_select_warp_width<384>(selected,weights,probs,bias,resident,lambda,used,scale,ram_cost);
    else ds4_cache_affinity_select_warp_width<256>(selected,weights,probs,bias,resident,lambda,used,scale,ram_cost);
}
#endif
