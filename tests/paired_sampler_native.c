/* CPU-only check of the actual Mac request parser. Never opens a model. */
#define DS4_SERVER_TEST 1
#define DS4_SERVER_TEST_NO_MAIN 1
/* Only tokenization is stubbed: exercising the native JSON parser must not
 * require a second GGUF mapping alongside the owner's active engine. */
#define ds4_tokenize_rendered_chat paired_test_tokenize
#include "../engines/ds4-mac/ds4_server.c"
#undef ds4_tokenize_rendered_chat
void paired_test_tokenize(ds4_engine *e, const char *text, ds4_tokens *out) {
    (void)e;(void)text;memset(out,0,sizeof(*out));
}

static bool requested_sampling(const request *r) {
    return r->temperature_set && fabsf(r->temperature-.83f)<1e-6f &&
           r->top_p_set && fabsf(r->top_p-.87f)<1e-6f &&
           r->top_k_set && r->top_k==37 &&
           r->min_p_set && fabsf(r->min_p-.06f)<1e-6f &&
           r->seed_set && r->seed==123456 && r->max_tokens==79 && r->paired.bias_set &&
           fabsf(r->paired.bias[DS4_QUALITY_REASONING]-.22f)<1e-6f &&
           fabsf(r->paired.presence-.14f)<1e-6f && fabsf(r->paired.frequency-.27f)<1e-6f &&
           fabsf(r->paired.repeat-1.12f)<1e-6f && r->paired.last_n==513 &&
           r->paired.bias_count==1 && r->paired.bias_tokens[0]==42 && fabsf(r->paired.bias_values[0]+.3f)<1e-6f &&
           r->paired.chain.order_set && r->paired.chain.order_count==5 &&
           fabsf(r->paired.chain.typical-.86f)<1e-6f && fabsf(r->paired.chain.dyn_range-.12f)<1e-6f &&
           fabsf(r->paired.chain.dyn_exponent-1.4f)<1e-6f && fabsf(r->paired.chain.xtc_probability-.2f)<1e-6f &&
           fabsf(r->paired.chain.xtc_threshold-.12f)<1e-6f && fabsf(r->paired.chain.dry_multiplier-.7f)<1e-6f &&
           r->paired.chain.dry_base==1.75f && r->paired.chain.dry_allowed_length==2 && r->paired.chain.dry_last_n==-1;
}

int main(int argc,char **argv) {
    request r;
    char err[256]={0};
    const char *body="{\"messages\":[{\"role\":\"user\",\"content\":\"q\"}],"
        "\"temperature\":0.83,\"top_p\":0.87,\"top_k\":37,\"min_p\":0.06,"
        "\"seed\":123456,\"max_tokens\":79,\"thinking\":true,\"quality_speed\":0.22,"
        "\"presence_penalty\":0.14,\"frequency_penalty\":0.27,\"repeat_penalty\":1.12,"
        "\"repeat_last_n\":513,\"logit_bias\":{\"42\":-0.3},"
        "\"samplers\":[\"top_k\",\"typ_p\",\"top_p\",\"min_p\",\"temperature\"],"
        "\"typ_p\":0.86,\"dynatemp_range\":0.12,\"dynatemp_exponent\":1.4,"
        "\"xtc_probability\":0.2,\"xtc_threshold\":0.12,\"dry_multiplier\":0.7,"
        "\"dry_base\":1.75,\"dry_allowed_length\":2,\"dry_penalty_last_n\":-1}";
    if (!parse_chat_request(NULL,NULL,body,128,1000001,&r,err,sizeof(err))) {
        fprintf(stderr,"Mac parser refused non-default samplers: %s\n",err);return 1;
    }
    if (!requested_sampling(&r)) {
        fprintf(stderr,"native values temp=%g top_p=%g top_k=%d min_p=%g seed=%llu max=%d bias=%g presence=%g frequency=%g repeat=%g last=%d bias_count=%u order_set=%d order_count=%u typ=%g dyn=%g exp=%g xtc=%g threshold=%g dry=%g base=%g allowed=%d dry_last=%d\n",
            r.temperature,r.top_p,r.top_k,r.min_p,(unsigned long long)r.seed,r.max_tokens,r.paired.bias[0],r.paired.presence,r.paired.frequency,r.paired.repeat,r.paired.last_n,r.paired.bias_count,
            r.paired.chain.order_set,r.paired.chain.order_count,r.paired.chain.typical,r.paired.chain.dyn_range,r.paired.chain.dyn_exponent,r.paired.chain.xtc_probability,r.paired.chain.xtc_threshold,
            r.paired.chain.dry_multiplier,r.paired.chain.dry_base,r.paired.chain.dry_allowed_length,r.paired.chain.dry_last_n);
        request_free(&r);return 2;
    }
    /* The same assertion must fail if a future proxy/parser drops to greedy. */
    r.temperature=0;r.top_p=1;r.top_k=0;r.min_p=0;
    if (requested_sampling(&r)) {request_free(&r);return 3;}
    request_free(&r);
    if(!parse_chat_request(NULL,NULL,"{\"messages\":[{\"role\":\"user\",\"content\":\"q\"}],\"seed\":0}",128,1000001,&r,err,sizeof(err)) || !r.seed_set || r.seed!=0)return 15;
    request_free(&r);
    /* Exact transform of a small independent logit vector, including a repeated
     * negative logit, a repeated positive one and an absent biased token. */
    ds4_paired_sampling knobs={.presence=.5f,.frequency=.25f,.repeat=2,.last_n=-1,
        .bias_count=1,.bias_tokens={3},.bias_values={1.5f}};
    float scores[]={4,-2,1,0};int history[]={0,1,1};unsigned counts[4];
    ds4_paired_logits(&knobs,scores,4,history,3,counts);
    if(fabsf(scores[0]-1.25f)>1e-6f || fabsf(scores[1]+5)>1e-6f || scores[2]!=1 || scores[3]!=1.5f)return 4;
    const char *phase="{\"reasoning\":{\"bias\":1.36125,\"temperature\":1,\"top_p\":0.95,\"top_k\":0,\"min_p\":0},"
        "\"answer\":{\"bias\":0.35,\"temperature\":0.6,\"top_p\":0.95,\"top_k\":0,\"min_p\":0},"
        "\"tool_call\":{\"bias\":0,\"temperature\":0.1,\"top_p\":0.95,\"top_k\":0,\"min_p\":0},"
        "\"code\":{\"bias\":0,\"temperature\":0.2,\"top_p\":0.95,\"top_k\":0,\"min_p\":0}}";
    ds4_paired_sampling parsed={0};
    if(!ds4_paired_parse("slstack_phase_samplers",&phase,&parsed)||!parsed.per_phase||
       parsed.phases[DS4_QUALITY_REASONING].temperature!=1 ||
       fabsf(parsed.phases[DS4_QUALITY_ANSWER].temperature-.6f)>1e-6f ||
       fabsf(parsed.phases[DS4_QUALITY_CODE].temperature-.2f)>1e-6f)return 5;
    ds4_chain_config cfg={.enabled=true,.order_set=true,.order_count=5,
        .order={DS4_CHAIN_TOP_K,DS4_CHAIN_TYPICAL,DS4_CHAIN_TOP_P,DS4_CHAIN_MIN_P,DS4_CHAIN_TEMP},
        .typical=1,.dyn_exponent=1,.dry_base=1.75f,.dry_allowed_length=2,.dry_last_n=-1};
    void *chain=ds4_chain_create(&cfg,NULL,NULL,0,123456,err,sizeof(err));
    float ordered[]={4,3,2,1};uint64_t seed=123456;
    if(!chain)return 6;
    int selected=ds4_chain_sample(chain,ordered,4,.5f,3,.85f,.1f,&seed);
    ds4_chain_free(chain);
    if(selected<0||selected>1||ordered[0]!=8||ordered[1]!=6||isfinite(ordered[2])||isfinite(ordered[3]))return 7;
    if(server_decode_budget(-1,0,1234)!=1234)return 8;
#if DS4_NATIVE_FRONTIER_ENABLE && defined(__APPLE__)
    if(argc>1){
        int ids[]={11,129264,12};ds4_tokens tokens={ids,3,3};float embedding[5120];
        for(unsigned i=0;i<5120;i++)embedding[i]=.25f;
        ds4_vision_span image={.token_start=1,.embedding={.data=embedding,.token_count=1}};
        memset(image.embedding.fingerprint,'a',32);
        ds4_ns_vision_body wire={0};
        if(!ds4_ns_vision_pack(&tokens,&image,1,&wire))return 9;
        FILE *output=fopen(argv[1],"wb");if(!output)return 10;
        bool written=fwrite(wire.data,1,wire.bytes,output)==wire.bytes&&!fclose(output);
        free(wire.data);if(!written)return 11;
    }
#else
    (void)argc;(void)argv;
#endif
    if(argc>2){
        char json[65536];FILE *input=fopen(argv[2],"rb");if(!input)return 12;
        size_t n=fread(json,1,sizeof(json)-1,input);bool end=feof(input);fclose(input);json[n]=0;
        if(!end || !parse_chat_request(NULL,NULL,json,128,1000001,&r,err,sizeof(err)))return 13;
        bool exact=r.temperature_set&&fabsf(r.temperature-.83f)<1e-6f&&r.top_p_set&&fabsf(r.top_p-.87f)<1e-6f&&
            r.top_k_set&&r.top_k==37&&r.min_p_set&&fabsf(r.min_p-.06f)<1e-6f&&r.paired.bias_set&&
            fabsf(r.paired.bias[0]-.22f)<1e-6f&&r.max_tokens==-1&&r.paired.chain.order_count==5&&r.paired.chain.dry_last_n==-1;
        request_free(&r);if(!exact)return 14;
        puts("PASS actual browser -> paired relay -> native Mac JSON parser; nondefault samplers preserved");
    }
    puts("PASS Mac native sampler values, penalties/logit-bias transform, four phase profiles and greedy negative control; no model opened");
    return 0;
}
