/* Previous-final-logit margin policy; independent of cache and routing. */
#ifndef DS4_CACHE_AFFINITY_ADAPTIVE_H
#define DS4_CACHE_AFFINITY_ADAPTIVE_H
#include <math.h>
#include <stddef.h>
#include <stdint.h>
static inline int ds4_cache_affinity_adaptive_mean(
        const float *logits, uint32_t count, float minimum, float maximum,
        float hard_gap, float easy_gap, float *mean, float *margin) {
    if(!mean || !margin || !isfinite(minimum) || !isfinite(maximum) ||
       minimum<0.0f || maximum>3.0f || minimum>maximum ||
       !isfinite(hard_gap) || !isfinite(easy_gap) ||
       hard_gap<0.0f || easy_gap<=hard_gap) return 0;
    /* Equal bounds are a fixed policy and require no vocabulary scan. */
    if(minimum==maximum){*mean=minimum;*margin=0.0f;return 1;}
    if(!logits || count<2) return 0;
    float first=-INFINITY,second=-INFINITY;uint32_t finite_count=0;
    for(uint32_t i=0;i<count;i++) {
        const float value=logits[i];
        if(isnan(value) || value==INFINITY) return 0;
        if(value==-INFINITY) continue;
        finite_count++;
        if(value>first){second=first;first=value;}
        else if(value>second)second=value;
    }
    if(finite_count<2) return 0;
    const float gap=first-second;
    float fraction=gap<=hard_gap?0.0f:gap>=easy_gap?1.0f:
        (gap-hard_gap)/(easy_gap-hard_gap);
    *margin=gap;
    *mean=minimum+(maximum-minimum)*fraction;
    return 1;
}
#endif
