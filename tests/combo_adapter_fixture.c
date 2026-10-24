/* Execute the real adapter against bounded host tensors. This checks callback
 * safety/transaction semantics, never substitutes for actual CUDA parity. */
#include "ds4_native_session.h"
#include <assert.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

typedef struct {uint8_t*data;uint64_t bytes;} ds4_gpu_tensor;
typedef struct {
    uint32_t ctx,pos,tp_world,state_generation;bool valid;void*draft;
    ds4_gpu_tensor *window[40],*compressed[4],*index_cache[4],*previous_kv[4],*previous_score[4];
    uint32_t token_map[129280];struct {int32_t tail[3];}history;
}ds41_gpu_graph;
struct ds4_engine {ds4_backend backend;struct{bool active;}tp;};
struct ds4_session {
    ds4_engine*engine;bool distributed,ds41_graph_ready,checkpoint_valid,mtp_draft_valid;
    ds41_gpu_graph ds41_graph;ds4_tokens checkpoint;float*logits;
    int(*cancel)(void*);void*cancel_ud;
};
static int payload_set_err(char*err,size_t cap,const char*text){if(err&&cap)snprintf(err,cap,"%s",text);return 1;}
static bool ds4_session_is_ds41(ds4_session*s){return s!=NULL;}
static int ds4_gpu_synchronize(void){return 1;}
static int ds4_gpu_tensor_read(ds4_gpu_tensor*t,uint64_t off,void*out,size_t n){
    if(off>t->bytes||n>t->bytes-off)return 0;memcpy(out,t->data+off,n);return 1;
}
static int ds4_gpu_tensor_write(ds4_gpu_tensor*t,uint64_t off,const void*in,size_t n){
    if(off>t->bytes||n>t->bytes-off)return 0;memcpy(t->data+off,in,n);return 1;
}
static void ds41_graph_reset(ds41_gpu_graph*g){g->pos=0;g->valid=true;++g->state_generation;memset(&g->history,0,sizeof(g->history));}
void ds4_tokens_free(ds4_tokens*t){free(t->v);memset(t,0,sizeof(*t));}
void ds4_session_rewind(ds4_session*s,int pos){s->checkpoint.len=pos;s->checkpoint_valid=false;}
void ds4_session_invalidate(ds4_session*s){s->checkpoint_valid=false;s->checkpoint.len=0;ds41_graph_reset(&s->ds41_graph);}
#define DS4_HAS_DEEPSEEK41_GPU 1
#include "ds4_native_spans.inc"
#include "ds4_native_session.inc"

typedef struct {uint8_t*data;uint64_t bytes,cursor;int32_t*tokens;float*logits;dsv41_c_span owners[55];uint32_t count,pos;unsigned reads;}snapshot;
static int take(void*ud,const ds4_native_view*v,char*err,size_t cap){
    snapshot*out=ud;(void)err;(void)cap;out->pos=v->position;out->count=v->span_count;
    memcpy(out->owners,v->spans,v->span_count*sizeof(*v->spans));
    out->tokens=malloc((size_t)v->position*4);out->logits=malloc(129280*4);
    memcpy(out->tokens,v->tokens,(size_t)v->position*4);memcpy(out->logits,v->logits,129280*4);
    for(uint32_t i=0;i<v->span_count;++i)out->bytes+=v->spans[i].bytes;
    out->data=malloc((size_t)out->bytes);assert(out->data&&out->tokens&&out->logits);
    uint64_t dst=0;
    for(uint32_t i=0;i<v->span_count;++i)for(uint64_t off=0;off<v->spans[i].bytes;){
        size_t n=(size_t)(v->spans[i].bytes-off);if(n>65536)n=65536;
        assert(v->read_span(v->state,i,off,out->data+dst,n));dst+=n;off+=n;
    }
    assert(!v->read_span(v->state,v->span_count,0,out->data,4));
    assert(!v->read_span(v->state,0,0,out->data,65537));return 1;
}
static int same(void*ud,const ds4_native_view*v,char*err,size_t cap){
    snapshot*want=ud;(void)err;(void)cap;uint64_t src=0;uint8_t page[65536];
    if(want->pos!=v->position||want->count!=v->span_count||memcmp(want->tokens,v->tokens,(size_t)v->position*4)||memcmp(want->logits,v->logits,129280*4))return 0;
    for(uint32_t i=0;i<v->span_count;++i){
        if(want->owners[i].kind!=v->spans[i].kind||want->owners[i].index!=v->spans[i].index||want->owners[i].bytes!=v->spans[i].bytes)return 0;
        for(uint64_t off=0;off<v->spans[i].bytes;){size_t n=(size_t)(v->spans[i].bytes-off);if(n>65536)n=65536;
            if(!v->read_span(v->state,i,off,page,n)||memcmp(page,want->data+src,n))return 0;
            src+=n;off+=n;
        }
    }return 1;
}
static int pull(void*ud,void*dst,size_t n){snapshot*s=ud;assert(n<=65536);++s->reads;
    if(s->cursor>s->bytes||n>s->bytes-s->cursor)return 0;memcpy(dst,s->data+s->cursor,n);s->cursor+=n;return 1;
}
static int reject(void*ud,const ds4_native_view*v,char*err,size_t cap){(void)ud;(void)v;(void)err;(void)cap;return 0;}
static int cancelled(void*ud){(void)ud;return 1;}
static ds4_gpu_tensor* tensor(size_t bytes,unsigned seed){ds4_gpu_tensor*t=malloc(sizeof(*t));assert(t);t->bytes=bytes;t->data=malloc(bytes);assert(t->data);for(size_t i=0;i<bytes;++i)t->data[i]=(uint8_t)(i+seed);return t;}
int main(void){
    ds4_engine engine={DS4_BACKEND_CUDA,{false}};ds4_session*s=calloc(1,sizeof(*s));assert(s);
    s->engine=&engine;s->ds41_graph_ready=s->checkpoint_valid=true;s->logits=calloc(129280,4);assert(s->logits);
    ds41_gpu_graph*g=&s->ds41_graph;g->ctx=1024;g->pos=129;g->tp_world=1;g->valid=true;
    s->checkpoint.v=malloc(129*4);s->checkpoint.len=s->checkpoint.cap=129;
    for(int i=0;i<129;++i)s->checkpoint.v[i]=i;
    for(int i=0;i<129280;++i)g->token_map[i]=i+17;
    for(int i=0;i<40;++i)g->window[i]=tensor(128*512*4,i);
    for(int i=0;i<4;++i){g->compressed[i]=tensor(1024*512*4,40+i);g->index_cache[i]=tensor(1024*128*4,50+i);g->previous_kv[i]=tensor(512*4,60+i);g->previous_score[i]=tensor(512*4,70+i);}
    char err[256]={0};snapshot initial={0};assert(!ds4_session_native_visit(s,take,&initial,err,sizeof(err)));assert(initial.count==54);
    memset(g->window[0]->data,0,128*512*4);s->checkpoint_valid=false;
    assert(!ds4_session_native_restore(s,initial.pos,initial.tokens,initial.logits,pull,&initial,same,&initial,err,sizeof(err)));
    assert(initial.cursor==initial.bytes&&s->checkpoint_valid&&g->valid&&g->pos==129&&g->history.tail[0]==145);
    assert(!ds4_session_native_visit(s,same,&initial,err,sizeof(err)));
    /* Wrong target refuses after writes and never publishes. */
    initial.cursor=0;assert(ds4_session_native_restore(s,129,initial.tokens,initial.logits,pull,&initial,reject,NULL,err,sizeof(err)));
    assert(!s->checkpoint_valid&&!g->valid);
    initial.cursor=0;assert(!ds4_session_native_restore(s,129,initial.tokens,initial.logits,pull,&initial,same,&initial,err,sizeof(err)));
    /* Reject stale base before consuming any delta bytes. */
    int32_t added=130;initial.reads=0;assert(ds4_session_native_apply_delta(s,129,130,&added,initial.logits,pull,&initial,reject,NULL,same,&initial,err,sizeof(err)));
    assert(initial.reads==0&&s->checkpoint_valid&&g->valid);
    /* Positive delta writes the actual production planner's ranges, including
     * compressed append rows, then verifies the complete target owners. */
    dsv41_c_plan plan;assert(dsv41_c_delta_mode(1,1024,129,130,&plan));
    ds41_state_span next_spans[55];assert(ds41_state_spans(g,130,next_spans)==plan.span_count);
    s->checkpoint.v=realloc(s->checkpoint.v,130*4);assert(s->checkpoint.v);
    s->checkpoint.v[129]=added;s->checkpoint.len=s->checkpoint.cap=130;g->pos=130;
    for(uint32_t i=0;i<plan.slice_count;++i){dsv41_c_slice*r=&plan.slices[i];
        for(uint64_t j=0;j<r->bytes;++j)next_spans[r->span].tensor->data[r->offset+j]^=0x5a;}
    snapshot target={0};assert(!ds4_session_native_visit(s,take,&target,err,sizeof(err)));
    snapshot delta={0};delta.bytes=plan.state_bytes;delta.data=malloc((size_t)delta.bytes);assert(delta.data);
    uint64_t offsets[55],offset=0,dst=0;
    for(uint32_t i=0;i<plan.span_count;++i){offsets[i]=offset;offset+=plan.spans[i].bytes;}
    for(uint32_t i=0;i<plan.slice_count;++i){dsv41_c_slice*r=&plan.slices[i];memcpy(delta.data+dst,target.data+offsets[r->span]+r->offset,(size_t)r->bytes);dst+=r->bytes;}
    assert(dst==delta.bytes);initial.cursor=0;
    assert(!ds4_session_native_restore(s,129,initial.tokens,initial.logits,pull,&initial,same,&initial,err,sizeof(err)));
    assert(!ds4_session_native_apply_delta(s,129,130,&added,target.logits,pull,&delta,same,&initial,same,&target,err,sizeof(err)));
    assert(delta.cursor==delta.bytes&&g->pos==130&&s->checkpoint_valid);
    assert(!ds4_session_native_visit(s,same,&target,err,sizeof(err)));
    initial.cursor=0;assert(!ds4_session_native_restore(s,129,initial.tokens,initial.logits,pull,&initial,same,&initial,err,sizeof(err)));
    /* Saved prefix rewind restores only mutable rings/tails. */
    snapshot prefix={0};g->pos=127;s->checkpoint.len=127;
    assert(!ds4_session_native_visit(s,take,&prefix,err,sizeof(err)));
    g->pos=129;s->checkpoint.len=129;memset(g->window[1]->data,0,128*512*4);
    snapshot mutable={0};mutable.bytes=0;
    for(uint32_t i=0;i<prefix.count;++i)if(prefix.owners[i].kind!=DSV41_C_COMPRESSED&&prefix.owners[i].kind!=DSV41_C_INDEX)mutable.bytes+=prefix.owners[i].bytes;
    mutable.data=malloc((size_t)mutable.bytes);uint64_t from=0,to=0;
    for(uint32_t i=0;i<prefix.count;++i){size_t n=(size_t)prefix.owners[i].bytes;
        if(prefix.owners[i].kind!=DSV41_C_COMPRESSED&&prefix.owners[i].kind!=DSV41_C_INDEX){memcpy(mutable.data+to,prefix.data+from,n);to+=n;}from+=n;}
    /* Base verifier is intentionally permissive in this mechanical fixture;
     * the production hasher and real parity gate validate the complete base. */
    snapshot changed={0};assert(!ds4_session_native_visit(s,take,&changed,err,sizeof(err)));
    assert(!ds4_session_native_rewind(s,127,prefix.logits,pull,&mutable,same,&changed,same,&prefix,err,sizeof(err)));
    assert(s->checkpoint_valid&&g->pos==127&&mutable.cursor==mutable.bytes);
    assert(!ds4_session_native_visit(s,same,&prefix,err,sizeof(err)));
    /* Cancellation after admission invalidates partial restore. */
    s->cancel=cancelled;initial.cursor=0;
    assert(ds4_session_native_restore(s,129,initial.tokens,initial.logits,pull,&initial,same,&initial,err,sizeof(err)));
    assert(!s->checkpoint_valid&&!g->valid);s->cancel=NULL;
    puts("PASS: real adapter host fixture: restore, delta, owner verification, stale-base refusal, target refusal, mutable rewind, cancellation");
    return 0;
}
