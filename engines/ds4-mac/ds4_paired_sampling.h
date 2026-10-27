#ifndef DS4_PAIRED_SAMPLING_H
#define DS4_PAIRED_SAMPLING_H
/* Token phases are the same implementation used by the RTX native decoder.
 * Selection-only quality/speed semantics come from Studio's Mac MiMo
 * ds4_mimo_impl.h:mimo_route (100-token warmup, uncached-expert penalty).
 * Native-session ds4 uses fully resident Metal experts: every candidate is
 * cached, so that policy leaves router scores and combine weights unchanged.
 * Never reinterpret this control as temperature or a logit bias. */
#include "ds4_quality_phase.h"
#include "ds4_sampling_bridge.h"
typedef struct {float temperature,top_p,min_p,bias;int top_k;} ds4_paired_phase;
typedef struct {
    bool per_phase,bias_set,bias_map;
    ds4_paired_phase phases[DS4_QUALITY_PHASE_COUNT];
    float bias[DS4_QUALITY_PHASE_COUNT];
    float presence,frequency,repeat;
    int last_n;
    int bias_tokens[1024];float bias_values[1024];unsigned bias_count;
    ds4_chain_config chain;
} ds4_paired_sampling;
static int ds4_paired_phase_index(const char *name) {
    for(int i=0;i<DS4_QUALITY_PHASE_COUNT;i++)if(!strcmp(name,ds4_quality_phase_names[i]))return i;
    return -1;
}
static bool ds4_paired_number(const char **p,float *out,double lo,double hi) {
    double v=0;if(!json_number(p,&v)||!(v>=lo&&v<=hi))return false;
    *out=(float)v;return true;
}
static bool ds4_paired_int(const char **p,int *out,int lo,int hi) {
    double v=0;if(!json_number(p,&v)||!(v>=lo&&v<=hi)||floor(v)!=v)return false;
    *out=(int)v;return true;
}
static bool ds4_paired_phase_parse(const char **p,ds4_paired_phase *out) {
    unsigned seen=0;json_ws(p);if(*(*p)++!='{')return false;
    while(1){char *key=NULL;unsigned bit=0;bool ok=false;
        if(!json_string(p,&key))return false;json_ws(p);if(*(*p)++!=':'){free(key);return false;}
        if(!strcmp(key,"temperature")){bit=1;ok=ds4_paired_number(p,&out->temperature,0,2);}
        else if(!strcmp(key,"top_p")){bit=2;ok=ds4_paired_number(p,&out->top_p,.001,1);}
        else if(!strcmp(key,"min_p")){bit=4;ok=ds4_paired_number(p,&out->min_p,0,1);}
        else if(!strcmp(key,"bias")){bit=8;ok=ds4_paired_number(p,&out->bias,0,1.36125);}
        else if(!strcmp(key,"top_k")){bit=16;ok=ds4_paired_int(p,&out->top_k,0,1024);}
        free(key);if(!ok||(seen&bit))return false;seen|=bit;
        json_ws(p);if(**p=='}'){(*p)++;break;}if(*(*p)++!=',')return false;
    }return seen==31;
}
static bool ds4_paired_map(const char **p,ds4_paired_sampling *s,bool sampler) {
    unsigned seen=0;json_ws(p);if(*(*p)++!='{')return false;
    while(1){char *name=NULL;if(!json_string(p,&name))return false;
        int phase=ds4_paired_phase_index(name);free(name);json_ws(p);
        if(phase<0||(seen&(1u<<phase))||*(*p)++!=':')return false;seen|=1u<<phase;
        if(sampler){if(!ds4_paired_phase_parse(p,&s->phases[phase]))return false;}
        else if(!ds4_paired_number(p,&s->bias[phase],0,1.36125))return false;
        json_ws(p);if(**p=='}'){(*p)++;break;}if(*(*p)++!=',')return false;
    }return seen==15;
}
static bool ds4_paired_field(const char *key) {
    return !strcmp(key,"slstack_phase_samplers")||!strcmp(key,"quality_speed")||
        !strcmp(key,"presence_penalty")||!strcmp(key,"frequency_penalty")||
        !strcmp(key,"repeat_penalty")||!strcmp(key,"repetition_penalty")||
        !strcmp(key,"repeat_last_n")||!strcmp(key,"penalty_last_n")||!strcmp(key,"logit_bias")||
        !strcmp(key,"samplers")||!strcmp(key,"typ_p")||!strcmp(key,"typical_p")||
        !strcmp(key,"dynatemp_range")||!strcmp(key,"dynatemp_exponent")||
        !strcmp(key,"xtc_probability")||!strcmp(key,"xtc_threshold")||
        !strcmp(key,"dry_multiplier")||!strcmp(key,"dry_base")||!strcmp(key,"top_n_sigma")||
        !strcmp(key,"dry_allowed_length")||!strcmp(key,"dry_penalty_last_n");
}
static bool ds4_paired_parse(const char *key,const char **p,ds4_paired_sampling *s) {
    if(!strcmp(key,"samplers")) {
        static const char *names[]={"top_k","typ_p","top_p","min_p","temperature","xtc","dry","penalties","top_n_sigma"};
        s->chain.enabled=s->chain.order_set=true;json_ws(p);if(*(*p)++!='[')return false;json_ws(p);
        if(**p==']'){(*p)++;return true;}unsigned seen=0;
        while(1){char *name=NULL;int kind=-1;if(s->chain.order_count>=16||!json_string(p,&name))return false;
            for(int i=0;i<9;i++)if(!strcmp(name,names[i]))kind=i;free(name);
            if(kind<0||(seen&(1u<<kind)))return false;seen|=1u<<kind;s->chain.order[s->chain.order_count++]=kind;
            json_ws(p);if(**p==']'){(*p)++;break;}if(*(*p)++!=',')return false;}
        return true;
    }
    if(!strcmp(key,"typ_p")||!strcmp(key,"typical_p")){s->chain.enabled=true;return ds4_paired_number(p,&s->chain.typical,0,1);}
    if(!strcmp(key,"dynatemp_range")){s->chain.enabled=true;return ds4_paired_number(p,&s->chain.dyn_range,0,2);}
    if(!strcmp(key,"dynatemp_exponent"))return ds4_paired_number(p,&s->chain.dyn_exponent,.001,100);
    if(!strcmp(key,"xtc_probability")){s->chain.enabled=true;return ds4_paired_number(p,&s->chain.xtc_probability,0,1);}
    if(!strcmp(key,"xtc_threshold"))return ds4_paired_number(p,&s->chain.xtc_threshold,0,1);
    if(!strcmp(key,"dry_multiplier")){s->chain.enabled=true;return ds4_paired_number(p,&s->chain.dry_multiplier,0,100);}
    if(!strcmp(key,"dry_base"))return ds4_paired_number(p,&s->chain.dry_base,1,100);
    if(!strcmp(key,"top_n_sigma")){s->chain.enabled=true;return ds4_paired_number(p,&s->chain.top_n_sigma,-1,100);}
    if(!strcmp(key,"dry_allowed_length"))return ds4_paired_int(p,&s->chain.dry_allowed_length,0,1048576);
    if(!strcmp(key,"dry_penalty_last_n"))return ds4_paired_int(p,&s->chain.dry_last_n,-1,1048576);
    if(!strcmp(key,"slstack_phase_samplers")){s->per_phase=true;return ds4_paired_map(p,s,true);}
    if(!strcmp(key,"quality_speed")){
        s->bias_set=true;json_ws(p);s->bias_map=**p=='{';
        if(s->bias_map)return ds4_paired_map(p,s,false);
        if(!ds4_paired_number(p,&s->bias[0],0,1.36125))return false;
        for(int i=1;i<DS4_QUALITY_PHASE_COUNT;i++)s->bias[i]=s->bias[0];return true;
    }
    if(!strcmp(key,"presence_penalty"))return ds4_paired_number(p,&s->presence,-2,2);
    if(!strcmp(key,"frequency_penalty"))return ds4_paired_number(p,&s->frequency,-2,2);
    if(!strcmp(key,"repeat_penalty")||!strcmp(key,"repetition_penalty"))return ds4_paired_number(p,&s->repeat,.001,100);
    if(!strcmp(key,"repeat_last_n")||!strcmp(key,"penalty_last_n"))return ds4_paired_int(p,&s->last_n,-1,1048576);
    if(!strcmp(key,"logit_bias")){
        json_ws(p);if(*(*p)++!='{')return false;json_ws(p);if(**p=='}'){(*p)++;return true;}
        while(1){char *name=NULL,*end=NULL;if(s->bias_count>=1024||!json_string(p,&name))return false;
            long token=strtol(name,&end,10);bool ok=*name&&!*end&&token>=0&&token<129280;free(name);
            json_ws(p);if(!ok||*(*p)++!=':')return false;
            for(unsigned i=0;i<s->bias_count;i++)if(s->bias_tokens[i]==token)return false;
            s->bias_tokens[s->bias_count]=(int)token;
            if(!ds4_paired_number(p,&s->bias_values[s->bias_count],-100,100))return false;s->bias_count++;
            json_ws(p);if(**p=='}'){(*p)++;break;}if(*(*p)++!=',')return false;
        }return true;
    }return false;
}
static bool ds4_paired_logits_active(const ds4_paired_sampling *s) {
    return s->presence!=0||s->frequency!=0||s->repeat!=1||s->bias_count!=0;
}
static void ds4_paired_logits(const ds4_paired_sampling *s,float *logits,int vocab,
        const int *tokens,int count,unsigned *counts) {
    memset(counts,0,(size_t)vocab*sizeof(*counts));
    int begin=s->last_n<0?0:count-s->last_n;if(begin<0)begin=0;
    for(int i=begin;i<count;i++)if(tokens[i]>=0&&tokens[i]<vocab)counts[tokens[i]]++;
    for(int i=0;i<vocab;i++)if(counts[i]){
        logits[i]=logits[i]>0?logits[i]/s->repeat:logits[i]*s->repeat;
        logits[i]-=s->presence+s->frequency*counts[i];
    }
    for(unsigned i=0;i<s->bias_count;i++)if(s->bias_tokens[i]<vocab)logits[s->bias_tokens[i]]+=s->bias_values[i];
}
#endif
