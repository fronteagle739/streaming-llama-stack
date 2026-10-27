#ifndef DS4_SAMPLING_BRIDGE_H
#define DS4_SAMPLING_BRIDGE_H
#include <stdint.h>
#include <stdbool.h>
enum {DS4_CHAIN_TOP_K,DS4_CHAIN_TYPICAL,DS4_CHAIN_TOP_P,DS4_CHAIN_MIN_P,DS4_CHAIN_TEMP,DS4_CHAIN_XTC,DS4_CHAIN_DRY,DS4_CHAIN_PENALTIES,DS4_CHAIN_SIGMA};
typedef struct {
    bool enabled,order_set;
    unsigned order_count;int order[16];
    float typical,dyn_range,dyn_exponent,xtc_probability,xtc_threshold;
    float dry_multiplier,dry_base,top_n_sigma;
    int dry_allowed_length,dry_last_n;
} ds4_chain_config;
#ifdef __cplusplus
extern "C" {
#endif
void *ds4_chain_create(const ds4_chain_config*,void *engine,const int *history,int count,uint64_t seed,char *err,unsigned cap);
int ds4_chain_sample(void*,float *logits,int vocab,float temperature,int top_k,float top_p,float min_p,uint64_t *rng);
void ds4_chain_accept(void*,int token);
void ds4_chain_free(void*);
#ifdef __cplusplus
}
#endif
#endif
