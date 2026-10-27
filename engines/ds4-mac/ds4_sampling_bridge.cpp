/* Reuse the retained llama.cpp sampler implementations verbatim. No second
 * model/vocabulary mapping: DRY breaker tokenization uses the resident ds4
 * engine. Only the final categorical draw uses ds4's existing64-bit RNG. */
#include "ds4_sampling_bridge.h"
extern "C" {
#include "ds4.h"
}
#include "llama-sampler.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
struct ds4_chain {
    struct node {int kind;llama_sampler *sampler;};
    ds4_chain_config cfg;
    std::vector<node> nodes;
    std::vector<llama_token_data> candidates;
    float temperature=-1,top_p=-1,min_p=-1;int top_k=-1;
    ~ds4_chain(){for(auto &node:nodes)llama_sampler_free(node.sampler);}
};
/* Same overlapping-breaker construction as llama-sampler.cpp, using ds4's
 * byte-exact vocabulary and tokenizer instead of opening a llama model. */
static std::vector<std::vector<llama_token>> breakers(ds4_engine *e) {
    std::vector<std::vector<llama_token>> result;
    for(int id=0;id<ds4_engine_vocab_size(e);id++){
        size_t n=0;char *raw=ds4_token_text(e,id,&n);std::string word(raw,n);free(raw);
        // The WebUI's default sequence breakers are single characters. Every
        // token containing one is a breaker, including multi-byte BPE pieces.
        if(word.find_first_of("\n:\"*")!=std::string::npos)result.push_back({id});
    }
    return result;
}
extern "C" void *ds4_chain_create(const ds4_chain_config *cfg,void *engine,const int *history,int count,uint64_t seed,char *err,unsigned cap) {
    try {
        std::unique_ptr<ds4_chain> c(new ds4_chain);c->cfg=*cfg;
        std::vector<int> order;
        if(cfg->order_set)order.assign(cfg->order,cfg->order+cfg->order_count);
        else order={DS4_CHAIN_TOP_K,DS4_CHAIN_TYPICAL,DS4_CHAIN_TOP_P,DS4_CHAIN_MIN_P,DS4_CHAIN_XTC,DS4_CHAIN_TEMP};
        if(cfg->dry_multiplier>0 && std::find(order.begin(),order.end(),DS4_CHAIN_DRY)==order.end())order.insert(order.begin(),DS4_CHAIN_DRY);
        for(int kind:order){
            llama_sampler *p=nullptr;
            if(kind==DS4_CHAIN_TYPICAL)p=llama_sampler_init_typical(cfg->typical,1);
            else if(kind==DS4_CHAIN_XTC){uint32_t folded=uint32_t(seed)^uint32_t(seed>>32);if(folded==LLAMA_DEFAULT_SEED)folded--;p=llama_sampler_init_xtc(cfg->xtc_probability,cfg->xtc_threshold,1,folded);}
            else if(kind==DS4_CHAIN_DRY && cfg->dry_multiplier>0)p=llama_sampler_init_dry_testing(1000001,cfg->dry_multiplier,cfg->dry_base,cfg->dry_allowed_length,cfg->dry_last_n,breakers(static_cast<ds4_engine*>(engine)));
            else if(kind==DS4_CHAIN_SIGMA)p=llama_sampler_init_top_n_sigma(cfg->top_n_sigma);
            // Core filters get their phase-specific values at the token boundary.
            c->nodes.push_back({kind,p});
        }
        for(int i=0;i<count;i++)for(auto &node:c->nodes)if(node.sampler)llama_sampler_accept(node.sampler,history[i]);
        return c.release();
    }catch(const std::exception &e){snprintf(err,cap,"native sampler setup: %s",e.what());return nullptr;}
}
extern "C" int ds4_chain_sample(void *opaque,float *logits,int vocab,float temperature,int top_k,float top_p,float min_p,uint64_t *rng) {
    auto *c=static_cast<ds4_chain*>(opaque);
    try{
        c->candidates.resize(vocab);
        for(int i=0;i<vocab;i++)c->candidates[i]={i,logits[i],0};
        llama_token_data_array data={c->candidates.data(),c->candidates.size(),-1,false};
        for(auto &node:c->nodes){
            bool changed=false;
            if(node.kind==DS4_CHAIN_TOP_K && top_k!=c->top_k){llama_sampler_free(node.sampler);node.sampler=llama_sampler_init_top_k(top_k);changed=true;}
            if(node.kind==DS4_CHAIN_TOP_P && top_p!=c->top_p){llama_sampler_free(node.sampler);node.sampler=llama_sampler_init_top_p(top_p,1);changed=true;}
            if(node.kind==DS4_CHAIN_MIN_P && min_p!=c->min_p){llama_sampler_free(node.sampler);node.sampler=llama_sampler_init_min_p(min_p,1);changed=true;}
            if(node.kind==DS4_CHAIN_TEMP && temperature!=c->temperature){llama_sampler_free(node.sampler);node.sampler=llama_sampler_init_temp_ext(temperature,c->cfg.dyn_range,c->cfg.dyn_exponent);changed=true;}
            (void)changed;if(node.sampler)llama_sampler_apply(node.sampler,&data);
        }
        c->top_k=top_k;c->top_p=top_p;c->min_p=min_p;c->temperature=temperature;
        std::fill(logits,logits+vocab,-std::numeric_limits<float>::infinity());
        for(size_t i=0;i<data.size;i++)logits[data.data[i].id]=data.data[i].logit;
        return ds4_sample_logits(logits,vocab,1,0,1,0,rng);
    }catch(...){return -1;}
}
extern "C" void ds4_chain_accept(void *opaque,int token) {
    if(auto *c=static_cast<ds4_chain*>(opaque))for(auto &node:c->nodes)if(node.sampler)llama_sampler_accept(node.sampler,token);
}
extern "C" void ds4_chain_free(void *opaque){delete static_cast<ds4_chain*>(opaque);}
