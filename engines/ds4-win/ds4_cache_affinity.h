/* Selection-only cache affinity. The original probabilities are never changed. */
#ifndef DS4_CACHE_AFFINITY_H
#define DS4_CACHE_AFFINITY_H
#include <stdint.h>
#include <math.h>
#ifdef __CUDACC__
#define DS4_AFFINITY_FN __host__ __device__
#else
#define DS4_AFFINITY_FN
#endif
static DS4_AFFINITY_FN void ds4_cache_affinity_select(
        int32_t *selected, float *weights, const float *probs,
        const float *bias, const uint8_t *resident, float lambda,
        uint32_t experts, uint32_t used, float scale) {
    for (uint32_t j = 0; j < used; j++) selected[j] = -1;
    for (uint32_t e = 0; e < experts; e++) {
        const float score = probs[e] + (bias ? bias[e] : 0.0f) +
                            -(resident[e] ? 0.0f : lambda);
        for (uint32_t j = 0; j < used; j++) {
            const int32_t old = selected[j];
            if (old < 0 || score > probs[old] + (bias ? bias[old] : 0.0f) +
                                  -(resident[old] ? 0.0f : lambda)) {
                for (uint32_t k = used - 1; k > j; k--) selected[k] = selected[k-1];
                selected[j] = (int32_t)e;
                break;
            }
        }
    }
    float sum = 0.0f;
    for (uint32_t j = 0; j < used; j++) { weights[j] = probs[selected[j]]; sum += weights[j]; }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (uint32_t j = 0; j < used; j++) weights[j] = weights[j] / sum * scale;
}
typedef struct ds4_affinity_confidence {
    float replacement, width, min_delta, max_delta;
    uint32_t protected_count;
} ds4_affinity_confidence;
/* selected/weights initially contain the native top-k. No new residency state.
 * Only resident experts outside that set are available replacements. A missing
 * expert ahead by >= the local top-1..top-k spread gets no penalty. */
static DS4_AFFINITY_FN void ds4_cache_affinity_select_confident(
        int32_t *selected, float *weights, const float *probs,
        const float *bias, const uint8_t *resident, float lambda,
        uint32_t experts, uint32_t used, float scale,
        ds4_affinity_confidence *metrics) {
    ds4_affinity_confidence m={0};
    if(metrics)*metrics=m;
    if(lambda<=0.0f || !used || experts>384u) return;
    uint32_t misses=0;
    for(uint32_t j=0;j<used;j++) misses+=!resident[selected[j]];
    if(!misses) return;
    int found=0;float alternative=0.0f;
    for(uint32_t e=0;e<experts;e++) {
        int native=0;
        for(uint32_t j=0;j<used;j++) native|=selected[j]==(int32_t)e;
        if(resident[e] && !native) {
            const float score=probs[e]+(bias?bias[e]:0.0f);
            if(!found || score>alternative)alternative=score;
            found=1;
        }
    }
    if(!found) {m.protected_count=misses;if(metrics)*metrics=m;return;}
    const float first=probs[selected[0]]+(bias?bias[selected[0]]:0.0f);
    const float last=probs[selected[used-1]]+(bias?bias[selected[used-1]]:0.0f);
    m.replacement=alternative;m.width=fmaxf(first-last,1e-8f);
    float selection_scores[384];
    for(uint32_t e=0;e<experts;e++) {
        const float score=probs[e]+(bias?bias[e]:0.0f);
        const float gap=fmaxf(score-alternative,0.0f);
        const float delta=resident[e]?0.0f:lambda*fmaxf(0.0f,1.0f-gap/m.width);
        selection_scores[e]=score-delta;
    }
    m.min_delta=lambda;
    for(uint32_t j=0;j<used;j++) if(!resident[selected[j]]) {
        const uint32_t e=(uint32_t)selected[j];
        const float score=probs[e]+(bias?bias[e]:0.0f);
        const float delta=lambda*fmaxf(0.0f,1.0f-fmaxf(score-alternative,0.0f)/m.width);
        m.min_delta=fminf(m.min_delta,delta);m.max_delta=fmaxf(m.max_delta,delta);
        m.protected_count+=delta==0.0f;
    }
    if(metrics)*metrics=m;
    if(m.max_delta==0.0f) return; /* Native IDs and combine bytes are already right. */
    /* Reuse the unchanged selector on corrected selection scores, then restore
     * original unbiased gate probabilities for the model's combine weights. */
    ds4_cache_affinity_select(selected,weights,selection_scores,(const float *)0,
        resident,0.0f,experts,used,scale);
    float sum=0.0f;
    for(uint32_t j=0;j<used;j++){weights[j]=probs[selected[j]];sum+=weights[j];}
    sum=fmaxf(sum,6.103515625e-5f);
    for(uint32_t j=0;j<used;j++)weights[j]=weights[j]/sum*scale;
}
typedef struct ds4_affinity_resident_decision {
    uint32_t resident_count, applied, protected_top, insufficient_resident;
    float alternative, width;
} ds4_affinity_resident_decision;
/* Verify-only hard policy. Protect the native set when its missing top choice
 * exceeds the best resident by the native top1..topk width. Otherwise select
 * only residents, provided a complete top-k exists. No invented experts or
 * partial normalization when the cache has fewer than k eligible entries. */
static DS4_AFFINITY_FN int ds4_cache_affinity_select_resident_confident(
        int32_t *selected,float *weights,const float *probs,const float *bias,
        const uint8_t *resident,uint32_t experts,uint32_t used,float scale,
        ds4_affinity_resident_decision *decision) {
    ds4_affinity_resident_decision d={0};
    if(!used || used>experts || experts>384u)return 0;
    const int32_t first=selected[0],last=selected[used-1];
    if(first<0 || last<0 || (uint32_t)first>=experts || (uint32_t)last>=experts)return 0;
    float best=-INFINITY;
    for(uint32_t e=0;e<experts;e++)if(resident[e]) {
        d.resident_count++;
        best=fmaxf(best,probs[e]+(bias?bias[e]:0.0f));
    }
    d.alternative=best;
    const float first_score=probs[first]+(bias?bias[first]:0.0f);
    d.width=fmaxf(first_score-(probs[last]+(bias?bias[last]:0.0f)),1e-8f);
    if(d.resident_count<used)d.insufficient_resident=1;
    else if(!resident[first] && first_score-best>=d.width)d.protected_top=1;
    else {
        float scores[384];
        for(uint32_t e=0;e<experts;e++)scores[e]=resident[e] ? probs[e]+(bias?bias[e]:0.0f) : -INFINITY;
        ds4_cache_affinity_select(selected,weights,scores,(const float *)0,resident,0,experts,used,scale);
        float sum=0;
        for(uint32_t j=0;j<used;j++){weights[j]=probs[selected[j]];sum+=weights[j];}
        sum=fmaxf(sum,6.103515625e-5f);
        for(uint32_t j=0;j<used;j++)weights[j]=weights[j]/sum*scale;
        d.applied=1;
    }
    if(decision)*decision=d;
    return 1;
}
#undef DS4_AFFINITY_FN
#endif
