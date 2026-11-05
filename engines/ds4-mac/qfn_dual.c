/* QFN raw-token benchmark: actual RTX payload restore, then ordinary native
 * sampling/MTP. No prompt replay is permitted after an imported checkpoint. */
#define _POSIX_C_SOURCE 200809L
#include "ds4.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <limits.h>

static double clock_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
static void fail(const char *why) { fprintf(stderr,"QfnPair FAILED: %s\n",why); exit(2); }
static int integer(const char *v) { char *end=NULL;errno=0;long n=strtol(v,&end,10);if(errno||!end||end==v||*end||n<INT_MIN||n>INT_MAX)fail("invalid integer flag");return (int)n; }
static uint64_t unsigned_integer(const char *v) { char *end=NULL;errno=0;unsigned long long n=strtoull(v,&end,10);if(errno||!end||end==v||*end||*v=='-')fail("invalid unsigned flag");return n; }
static float number(const char *v) { char *end=NULL;errno=0;float n=strtof(v,&end);if(errno||!end||end==v||*end||!isfinite(n))fail("invalid numerical flag");return n; }
static ds4_tokens read_tokens(const char *path) {
    ds4_tokens out={0}; FILE *f=fopen(path,"rb"); if(!f)fail("cannot open token file");
    unsigned char b[4];size_t n;
    while((n=fread(b,1,4,f))==4) {
        unsigned t=(unsigned)b[0]|(unsigned)b[1]<<8|(unsigned)b[2]<<16|(unsigned)b[3]<<24;
        if(t>2147483647u)fail("invalid token ID");ds4_tokens_push(&out,(int)t);
    }
    if(n||ferror(f)||fclose(f)||!out.len)fail("short/empty token file");return out;
}
static FILE *output(const char *prefix,const char *suffix) {
    char path[4096];if(snprintf(path,sizeof(path),"%s%s",prefix,suffix)>=(int)sizeof(path))fail("output path too long");
    FILE *f=fopen(path,"wbx");if(!f)fail("output must not already exist");return f;
}
int main(int argc,char **argv) {
    if(argc==4&&!strcmp(argv[1],"--validate-vision"))return ds4_qfn_validate_vision(argv[2],argv[3]);
    if(argc==3&&!strcmp(argv[1],"--validate-model"))return ds4_qfn_validate_model(argv[2]);
    if(argc==2&&!strcmp(argv[1],"--metadata-self-test"))return ds4_qfn_metadata_self_test();
    if(argc==3&&!strcmp(argv[1],"--ngram-self-test"))return ds4_qfn_ngram_self_test(argv[2]);
    const char *model=NULL,*tokens=NULL,*warm=NULL,*prefix=NULL,*payload=NULL,*forced=NULL,*vision=NULL;
    int ctx=0,generate=128,mtp=3,threads=8,chunk=256,top_k=20,framed=0,warm_weights=1;
    float temp=0,top_p=.95f,min_p=0;uint64_t seed=1,bytes=0;
    for(int i=1;i<argc;i++) {
        const char *k=argv[i];if(i+1>=argc)fail("missing flag value");const char *v=argv[++i];
        if(!strcmp(k,"--model"))model=v;else if(!strcmp(k,"--tokens"))tokens=v;
        else if(!strcmp(k,"--warmup-tokens"))warm=v;else if(!strcmp(k,"--out"))prefix=v;
        else if(!strcmp(k,"--payload"))payload=v;else if(!strcmp(k,"--payload-bytes"))bytes=unsigned_integer(v);
        else if(!strcmp(k,"--force-tokens"))forced=v;else if(!strcmp(k,"--vision"))vision=v;
        else if(!strcmp(k,"--ctx"))ctx=integer(v);else if(!strcmp(k,"--decode-tokens"))generate=integer(v);
        else if(!strcmp(k,"--mtp-draft"))mtp=integer(v);else if(!strcmp(k,"--threads"))threads=integer(v);
        else if(!strcmp(k,"--prefill-chunk"))chunk=integer(v);else if(!strcmp(k,"--temperature"))temp=number(v);
        else if(!strcmp(k,"--payload-framed"))framed=integer(v);else if(!strcmp(k,"--warm-weights"))warm_weights=integer(v);
        else if(!strcmp(k,"--top-k"))top_k=integer(v);else if(!strcmp(k,"--top-p"))top_p=number(v);
        else if(!strcmp(k,"--min-p"))min_p=number(v);else if(!strcmp(k,"--seed"))seed=unsigned_integer(v);
        else fail("unknown option; no field is ignored");
    }
    if(!model||!tokens||!warm||!prefix||ctx<=0||generate<=0||generate>4096||mtp<1||mtp>3||threads<1||chunk<1||
       !isfinite(temp)||temp<0||!isfinite(top_p)||top_p<=0||top_p>1||!isfinite(min_p)||min_p<0||min_p>1||
       top_k<0||!seed||(payload&&!bytes)||(!payload&&bytes)||(framed!=0&&framed!=1)||(warm_weights!=0&&warm_weights!=1))fail("invalid benchmark settings");
    ds4_engine_options options={0};options.model_path=model;options.vision_path=vision;
    options.backend=DS4_BACKEND_METAL;options.n_threads=threads;options.context_size=ctx;
    options.prefill_chunk=(uint32_t)chunk;options.mtp_draft_tokens=mtp;options.glm_mtp=mtp>1;
    options.qwen4_mtp_depth=mtp>1?mtp:0;
    if(mtp==3 && temp>0)fail("Unsupported operation: fixed depth 3 with exact sampled decoding is not implemented; choose depth 2");
    options.dspark_exact_sampling=true;options.power_percent=100;
    options.warm_weights=warm_weights!=0;
    ds4_engine *engine=NULL;if(ds4_engine_open(&engine,&options))fail("native engine open failed");
    if(!ds4_engine_is_qwen4(engine))fail("QFN family identity differs");
    ds4_session *session=NULL;if(ds4_session_create(&session,engine,ctx))fail("native session creation failed");
    char error[512]={0};ds4_tokens w=read_tokens(warm);
    if(ds4_session_sync(session,&w,error,sizeof(error)))fail(error);
    fprintf(stderr,"QfnPair WARMUP COMPLETE rows=%d; awaiting second prompt/payload\n",w.len);fflush(stderr);
    printf("{\"event\":\"warmup_complete\",\"tokens\":%d}\n",w.len);fflush(stdout);
    // The real prompt file is read only after the short warm-up has completed.
    ds4_tokens prompt=read_tokens(tokens);
    if(prompt.len+w.len>=ctx||prompt.len+generate+32>=ctx)fail("insufficient context budget");
    if(w.len==prompt.len&&!memcmp(w.v,prompt.v,(size_t)w.len*sizeof(int)))fail("warmup and measured prompt must differ");
    double start=clock_s();
    if(payload) {
        FILE *in=!strcmp(payload,"-")?stdin:fopen(payload,"rb");if(!in)fail("cannot open payload");
        if(ds4_session_load_payload(session,in,bytes,error,sizeof(error)))fail(error);
        if(framed&&fgetc(in)!='G')fail("payload authentication commit missing");
        if(in!=stdin&&fclose(in))fail("payload close failed");
        const ds4_tokens *live=ds4_session_tokens(session);
        if(!live||live->len!=prompt.len||memcmp(live->v,prompt.v,(size_t)prompt.len*sizeof(int)))fail("imported token identity differs; replay refused");
    } else if(ds4_session_sync(session,&prompt,error,sizeof(error)))fail(error);
    double prefill_s=clock_s()-start;
    if(ds4_session_pos(session)!=prompt.len)fail("native frontier differs");
    printf("{\"event\":\"imported\",\"seconds\":%.9f,\"rows\":%d}\n",prefill_s,prompt.len);fflush(stdout);
    int V=ds4_engine_vocab_size(engine);float *logits=malloc((size_t)V*sizeof(float));if(!logits)fail("logit staging allocation failed");
    int observed=0;
    FILE *ids=output(prefix,".tokens.le32"),*prose=output(prefix,".txt");
    start=clock_s();int generated=0,mtp_cycles=0,mtp_extra=0;bool stop=false;
    while(generated<generate) {
        int token=ds4_session_sample(session,temp,top_k,top_p,min_p,&seed);
        if(token<0)fail("sampling failed");if(ds4_token_is_stop(engine,token))break;
        int accepted[17],count=1;accepted[0]=token;
        if(mtp>1) {
            count=ds4_session_eval_speculative(session,token,generate-generated,ds4_token_eos(engine),
                temp,top_k,top_p,min_p,&seed,accepted,17,error,sizeof(error));
            if(count<=0)fail(error);
            mtp_cycles++;
        } else if(ds4_session_eval(session,token,error,sizeof(error)))fail(error);
        const int before=generated;
        for(int j=0;j<count;j++) {
            if(ds4_token_is_stop(engine,accepted[j])){stop=true;break;}
            unsigned t=(unsigned)accepted[j];unsigned char b[4]={t,t>>8,t>>16,t>>24};
            if(fwrite(b,1,4,ids)!=4)fail("token output failed");
            size_t n=0;char *s=ds4_token_text(engine,accepted[j],&n);if(s&&fwrite(s,1,n,prose)!=n)fail("text output failed");free(s);generated++;
        }
        if(mtp>1&&generated>before)mtp_extra+=generated-before-1;
        if(stop)break;
    }
    double decode_s=clock_s()-start;if(fclose(ids)||fclose(prose))fail("output close failed");
    printf("{\"event\":\"decoded\",\"tokens\":%d,\"seconds\":%.9f}\n",generated,decode_s);fflush(stdout);
    // Quality observations use the same initial prefix as the unbiased RTX
    // control. They never teacher-force the answer used for the recall proof.
    if(forced) {
        if(payload) {
            FILE *in=!strcmp(payload,"-")?stdin:fopen(payload,"rb");if(!in)fail("cannot reopen observation payload");
            if(ds4_session_load_payload(session,in,bytes,error,sizeof(error)))fail(error);
            if(framed&&fgetc(in)!='G')fail("observation payload authentication commit missing");
            if(in!=stdin&&fclose(in))fail("observation payload close failed");
        } else if(ds4_session_sync(session,&prompt,error,sizeof(error)))fail(error);
        ds4_tokens force=read_tokens(forced);if(force.len>4096)fail("observation row budget exceeded");
        FILE *f=output(prefix,".f32");
        for(int i=0;i<force.len;i++) {
            if(ds4_session_copy_logits(session,logits,V)!=V)fail("logit copy failed");
            for(int j=0;j<V;j++)if(!isfinite(logits[j]))fail("nonfinite native continuation");
            if(fwrite(logits,sizeof(float),(size_t)V,f)!=(size_t)V)fail("logit write failed");
            if(ds4_session_eval(session,force.v[i],error,sizeof(error)))fail(error);observed++;
        }
        if(fclose(f))fail("logit close failed");ds4_tokens_free(&force);
    }
    FILE *result=output(prefix,".json");
    fprintf(result,"{\"prompt_tokens\":%d,\"warmup_tokens\":%d,\"payload_loaded\":%s,\"import_or_prefill_seconds\":%.9f,"
        "\"decode_tokens\":%d,\"decode_seconds\":%.9f,\"decode_tok_s\":%.9f,\"observed_rows\":%d,\"vocab\":%d,\"kv_bits\":16,\"mtp_draft\":%d,\"mtp_cycles\":%d,\"mtp_extra_tokens\":%d}\n",
        prompt.len,w.len,payload?"true":"false",prefill_s,generated,decode_s,generated/fmax(decode_s,1e-9),observed,V,mtp,mtp_cycles,mtp_extra);
    if(fclose(result))fail("result close failed");
    puts("{\"event\":\"complete\"}");fflush(stdout);
    free(logits);ds4_tokens_free(&prompt);ds4_tokens_free(&w);ds4_session_free(session);ds4_engine_close(engine);return 0;
}
