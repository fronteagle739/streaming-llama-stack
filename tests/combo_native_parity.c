/* Real CUDA target-only qualification. The checkpoint, mutable journal and
 * 384 full reference logit vectors stay in explicitly locked physical RAM. */
#include "ds4_native_checkpoint.h"
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#define ROWS 384
#define CONTEXT 262144
static void die(const char*why){fprintf(stderr,"COMBO_PARITY_FAIL %s\n",why);fflush(stderr);exit(2);}
static void check(int rc,const char*why){if(rc)die(why);}
static double now(void){LARGE_INTEGER c,f;QueryPerformanceCounter(&c);QueryPerformanceFrequency(&f);return(double)c.QuadPart/f.QuadPart;}
static void*locked(size_t n){
    /* The outer guardian supplies total host admission. */
    void*p=VirtualAlloc(NULL,n,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE);if(!p)die("RAM allocation failed");
    SIZE_T lo=0,hi=0;if(!GetProcessWorkingSetSize(GetCurrentProcess(),&lo,&hi))die("working set query failed");
    if(!SetProcessWorkingSetSize(GetCurrentProcess(),lo+n+(64u<<20),hi+n+(128u<<20))||!VirtualLock(p,n))die("checkpoint/reference physical lock refused");
    return p;
}
typedef struct {
    ds4_native_checkpoint identity;uint8_t*data,*mutable;uint64_t bytes,cursor,mutable_bytes,mutable_cursor;
    int32_t*tokens;float*logits;
} saved_state;
static int save_state(void*ud,const ds4_native_view*v,char*err,size_t cap){
    saved_state*s=ud;if(!ds4_native_capture(&s->identity,v,err,cap))return 0;
    for(uint32_t i=0;i<v->span_count;++i){s->bytes+=v->spans[i].bytes;
        if(v->spans[i].kind!=DSV41_C_COMPRESSED&&v->spans[i].kind!=DSV41_C_INDEX)s->mutable_bytes+=v->spans[i].bytes;}
    s->data=locked((size_t)s->bytes+(size_t)v->position*4+129280*4);
    s->tokens=(int32_t*)(s->data+s->bytes);s->logits=(float*)(s->tokens+v->position);
    memcpy(s->tokens,v->tokens,(size_t)v->position*4);memcpy(s->logits,v->logits,129280*4);
    s->mutable=locked((size_t)s->mutable_bytes);uint64_t offset=0,m=0;
    for(uint32_t i=0;i<v->span_count;++i){bool mutable=v->spans[i].kind!=DSV41_C_COMPRESSED&&v->spans[i].kind!=DSV41_C_INDEX;
        for(uint64_t off=0;off<v->spans[i].bytes;){size_t n=(size_t)(v->spans[i].bytes-off);if(n>65536)n=65536;
            if(!v->read_span(v->state,i,off,s->data+offset,n))return 0;
            if(mutable){memcpy(s->mutable+m,s->data+offset,n);m+=n;}
            offset+=n;off+=n;
        }
    }return offset==s->bytes&&m==s->mutable_bytes;
}
static int read_full(void*ud,void*dst,size_t n){saved_state*s=ud;if(n>65536||s->cursor>s->bytes||n>s->bytes-s->cursor)return 0;memcpy(dst,s->data+s->cursor,n);s->cursor+=n;return 1;}
static int read_mutable(void*ud,void*dst,size_t n){saved_state*s=ud;if(n>65536||s->mutable_cursor>s->mutable_bytes||n>s->mutable_bytes-s->mutable_cursor)return 0;memcpy(dst,s->mutable+s->mutable_cursor,n);s->mutable_cursor+=n;return 1;}
static void restore(ds4_session*s,saved_state*base,char*err){base->cursor=0;check(ds4_session_native_restore(s,base->identity.position,base->tokens,base->logits,read_full,base,ds4_native_verify,&base->identity,err,512),err);if(base->cursor!=base->bytes)die("restore left owner bytes unread");}
int main(int argc,char**argv){
    if(argc!=3)die("expected model and proof-output path");
    ds4_engine_options opt={0};opt.model_path=argv[1];opt.backend=DS4_BACKEND_CUDA;opt.context_size=CONTEXT;
    opt.placement_ctx_hint=CONTEXT;opt.placement_session_count_hint=1;opt.request_sized_prefill_workspace=true;
    opt.n_threads=8;opt.ssd_streaming=true;opt.ssd_streaming_cache_bytes=24ull<<30;opt.cache_affinity_bias=0;opt.cache_affinity_bias_set=true;
    ds4_engine*e=NULL;ds4_session*s=NULL;char err[512]={0};check(ds4_engine_open(&e,&opt),"engine open");
    check(ds4_session_create(&s,e,CONTEXT),"session create");ds4_tokens prompt={0};
    ds4_encode_chat_prompt(e,NULL,"Explain how a rainbow forms, in clear English. Write three detailed paragraphs.",DS4_THINK_NONE,&prompt);
    check(ds4_session_sync(s,&prompt,err,sizeof(err)),err);saved_state base={0};
    check(ds4_session_native_visit(s,save_state,&base,err,sizeof(err)),err);
    float*reference=locked((size_t)ROWS*129280*4);float*row=locked(129280*4);int ids[ROWS];double timings[3]={0};
    for(int pass=0;pass<3;++pass){
        if(pass){restore(s,&base,err);
            if(pass==2){for(int i=0;i<3;++i)check(ds4_session_eval(s,ids[i],err,sizeof(err)),err);
                ds4_native_checkpoint current={0};check(ds4_session_native_visit(s,ds4_native_capture,&current,err,sizeof(err)),err);
                base.mutable_cursor=0;check(ds4_session_native_rewind(s,base.identity.position,base.logits,read_mutable,&base,ds4_native_verify,&current,ds4_native_verify,&base.identity,err,sizeof(err)),err);
                if(base.mutable_cursor!=base.mutable_bytes)die("rewind left mutable bytes unread");
            }
        }
        double start=now();
        for(int i=0;i<ROWS;++i){int token=ds4_session_argmax(s);
            if(ds4_session_copy_logits(s,row,129280)!=129280)die("logit copy");
            if(!pass){ids[i]=token;memcpy(reference+(size_t)i*129280,row,129280*4);}
            else if(token!=ids[i]||memcmp(reference+(size_t)i*129280,row,129280*4))die(pass==1?"round-trip full-vector/token mismatch":"rewind full-vector/token mismatch");
            check(ds4_session_eval(s,token,err,sizeof(err)),err);
        }
        timings[pass]=now()-start;fprintf(stderr,"COMBO_PARITY pass=%d vectors=384 byte_exact=384 tok_s=%.9f\n",pass,ROWS/timings[pass]);fflush(stderr);
        if(pass==0){ds4_session_free(s);s=NULL;ds4_engine_close(e);e=NULL;
            check(ds4_engine_open(&e,&opt),"reopen engine");check(ds4_session_create(&s,e,CONTEXT),"reopen session");}
    }
    ds4_session_free(s);ds4_engine_close(e);ds4_tokens_free(&prompt);
    char hash[65];if(!ds4_native_hash(reference,(size_t)ROWS*129280*4,hash))die("reference hash");
    FILE*f=fopen(argv[2],"wb");if(!f)die("proof open");
    fprintf(f,"{\"status\":\"PASS\",\"backend\":\"CUDA\",\"mode\":1,\"position\":%u,\"owners\":%u,\"vectors\":384,\"roundtrip_byte_exact\":384,\"rewind_byte_exact\":384,\"mean_kl\":0,\"top1\":1,\"reference_sha256\":\"%s\",\"payload_sha256\":\"%s\",\"seconds\":[%.9f,%.9f,%.9f],\"ultra_import_parity\":\"NOT_RUN\"}",base.identity.position,base.identity.owner_count,hash,base.identity.payload_sha256,timings[0],timings[1],timings[2]);
    if(fclose(f))die("proof close");puts("COMBO_PARITY_PASS native roundtrip and saved-frontier rewind; ULTRA parity still required");return 0;
}
