/* Compile the actual server executor excerpt with controlled backend calls.
 * This exercises pthread ownership and request/result transport, not a second
 * implementation of the scheduling algorithm. The runner pins the excerpt. */
#include "../engines/ds4-mac/ds4.h"
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef struct server server;
typedef struct server_slot server_slot;
typedef struct {
    int remaining,eos_token,top_k;
    float temperature,top_p,min_p;
    uint64_t rng;
    bool ignore_eos;
    ds4_think_mode think_mode;
} server_dspark_cycle;
struct ds4_session {int id,invalid,calls;};
struct server_slot {
    server *srv; ds4_session *session; atomic_bool cancelled;
    bool decode_pending,decode_in_flight,decode_done;
    int decode_token; bool decode_speculative,decode_dspark;
    server_dspark_cycle decode_cycle;
    int decode_accepted[17],decode_accepted_count,decode_rc;
    char decode_err[160];
};
struct server {
    bool batched_mode,model_stopping,model_busy;
    int slot_count,decode_pending,active_generations;
    pthread_mutex_t model_mu,inference_mu;
    pthread_cond_t model_cv;
    server_slot *slots;
};
static volatile sig_atomic_t g_stop_requested;
static server *current_server;
static int requested_rows=6;
static bool expect_batched;
static atomic_int dspark_calls,ignore_calls,plain_calls,qwen_calls;
static void *xmalloc(size_t n){void*p=malloc(n);assert(p);return p;}
static double now_sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void server_log(ds4_log_type type,const char *fmt,...){(void)type;(void)fmt;}
static bool slot_job_cancelled(const server_slot *slot){return atomic_load(&slot->cancelled);}
static void ownership(void){
    assert(pthread_mutex_trylock(&current_server->inference_mu)==EBUSY);
    if(expect_batched)assert(current_server->model_busy);
}
int ds4_session_eval(ds4_session *s,int token,char *err,size_t cap){
    (void)token;(void)err;(void)cap;ownership();s->calls++;atomic_fetch_add(&plain_calls,1);return 0;
}
int ds4_sessions_eval_batch(ds4_decode_item *items,int count,char *err,size_t cap){
    for(int i=0;i<count;i++)if(ds4_session_eval(items[i].session,items[i].token,err,cap))return 1;
    return 0;
}
int ds4_sessions_eval_batch_speculative_argmax(ds4_decode_item *items,int count,int (*accepted)[2],int *n,char *err,size_t cap){
    (void)err;(void)cap;ownership();
    for(int i=0;i<count;i++){accepted[i][0]=items[i].token;accepted[i][1]=items[i].token+1;n[i]=2;items[i].session->calls++;}
    atomic_fetch_add(&qwen_calls,count);return 0;
}
void ds4_session_invalidate(ds4_session*s){s->invalid++;}
static int emit(ds4_session*s,int token,int remaining,int*out,int cap){
    ownership();s->calls++;int n=requested_rows<remaining?requested_rows:remaining;
    for(int i=0;i<n && i<cap;i++)out[i]=token+i;
    return n;
}
int ds4_session_eval_speculative(ds4_session*s,int token,int remaining,int eos,
    float temp,int topk,float topp,float minp,uint64_t*rng,int*out,int cap,char*err,size_t errlen){
    (void)err;(void)errlen;
    assert(eos==2 && temp==.7f && topk==37 && topp==.93f && minp==.02f);
    assert(*rng==123);*rng=456;atomic_fetch_add(&dspark_calls,1);
    return emit(s,token,remaining,out,cap);
}
int ds4_session_eval_speculative_argmax_ignoring_eos(ds4_session*s,int token,int remaining,
    int eos,ds4_think_mode think,int*out,int cap,char*err,size_t errlen){
    (void)err;(void)errlen;assert(eos==2 && think==DS4_THINK_NONE);
    atomic_fetch_add(&ignore_calls,1);return emit(s,token,remaining,out,cap);
}
#include "native-server-decode-under-test.inc"

typedef struct {server*s;server_slot*slot;int kind,rc,n;server_dspark_cycle cycle;struct {int v[17];unsigned guard;} out;} call;
static server_dspark_cycle policy(bool ignore){
    server_dspark_cycle p={.remaining=128,.eos_token=2,.top_k=37,.temperature=.7f,
        .top_p=.93f,.min_p=.02f,.rng=123,.ignore_eos=ignore,.think_mode=DS4_THINK_NONE};return p;
}
static void *invoke(void*ptr){
    call*c=ptr;char err[160]={0};c->out.guard=0x5a1dcafeu;
    if(c->kind==0){c->rc=server_eval_token(c->s,c->slot,100,err,sizeof(err));c->n=c->rc?0:1;}
    else if(c->kind==1)c->rc=server_eval_tokens(c->s,c->slot,200,true,c->out.v,&c->n,err,sizeof(err));
    else c->rc=server_eval_tokens_impl(c->s,c->slot,300,false,&c->cycle,c->out.v,17,&c->n,err,sizeof(err));
    assert(c->out.guard==0x5a1dcafeu);return NULL;
}
static void setup(server*s,server_slot*slots,ds4_session*sessions,int count){
    memset(s,0,sizeof(*s));memset(slots,0,sizeof(*slots)*(size_t)count);memset(sessions,0,sizeof(*sessions)*(size_t)count);
    s->batched_mode=true;s->slot_count=count;s->active_generations=count;s->slots=slots;
    pthread_mutex_init(&s->model_mu,NULL);pthread_mutex_init(&s->inference_mu,NULL);pthread_cond_init(&s->model_cv,NULL);
    for(int i=0;i<count;i++){slots[i].srv=s;slots[i].session=&sessions[i];sessions[i].id=i;atomic_init(&slots[i].cancelled,false);}
    current_server=s;expect_batched=true;
}
static void finish(server*s,pthread_t worker){
    pthread_mutex_lock(&s->model_mu);s->model_stopping=true;pthread_cond_broadcast(&s->model_cv);pthread_mutex_unlock(&s->model_mu);
    pthread_join(worker,NULL);assert(!s->decode_pending && !s->model_busy);
    pthread_mutex_destroy(&s->model_mu);pthread_mutex_destroy(&s->inference_mu);pthread_cond_destroy(&s->model_cv);
}
static void mixed(void){
    server s;server_slot slots[3];ds4_session sessions[3];setup(&s,slots,sessions,3);
    pthread_t worker,threads[3];pthread_create(&worker,NULL,decode_worker_main,&s);
    call c[3]={0};for(int i=0;i<3;i++){c[i].s=&s;c[i].slot=&slots[i];c[i].kind=i;c[i].cycle=policy(false);pthread_create(&threads[i],NULL,invoke,&c[i]);}
    for(int i=0;i<3;i++){pthread_join(threads[i],NULL);assert(!c[i].rc && !sessions[i].invalid);}
    assert(c[0].n==1 && c[1].n==2 && c[2].n==6 && c[2].cycle.rng==456);
    for(int i=0;i<6;i++)assert(c[2].out.v[i]==300+i);
    assert(atomic_load(&plain_calls)==1 && atomic_load(&qwen_calls)==1 && atomic_load(&dspark_calls)==1);
    finish(&s,worker);
}
static void bounds_and_ignore(void){
    server s;server_slot slots[1];ds4_session sessions[1];setup(&s,slots,sessions,1);
    pthread_t worker;pthread_create(&worker,NULL,decode_worker_main,&s);
    call c={.s=&s,.slot=slots,.kind=2,.cycle=policy(true)};c.cycle.remaining=3;
    invoke(&c);assert(!c.rc && c.n==3 && c.cycle.rng==123 && atomic_load(&ignore_calls)==1);
    requested_rows=18;c.cycle=policy(false);invoke(&c);assert(c.rc && sessions[0].invalid==1);
    requested_rows=6;finish(&s,worker);
}
static void cancelled_pending(void){
    server s;server_slot slots[1];ds4_session sessions[1];setup(&s,slots,sessions,1);s.model_busy=true;
    call c={.s=&s,.slot=slots,.kind=2,.cycle=policy(false)};pthread_t thread;pthread_create(&thread,NULL,invoke,&c);
    pthread_mutex_lock(&s.model_mu);
    while(!slots[0].decode_pending)pthread_cond_wait(&s.model_cv,&s.model_mu);
    atomic_store(&slots[0].cancelled,true);assert(server_cancel_pending_decode_locked(&s,&slots[0]));pthread_mutex_unlock(&s.model_mu);
    pthread_join(thread,NULL);assert(c.rc==DS4_SESSION_SYNC_INTERRUPTED && sessions[0].calls==0 && s.decode_pending==0);
    s.model_busy=false;pthread_mutex_destroy(&s.model_mu);pthread_mutex_destroy(&s.inference_mu);pthread_cond_destroy(&s.model_cv);
}
static void unbatched(void){
    server s;server_slot slots[1];ds4_session sessions[1];setup(&s,slots,sessions,1);s.batched_mode=false;expect_batched=false;
    call c={.s=&s,.slot=slots,.kind=2,.cycle=policy(false)};invoke(&c);assert(!c.rc && c.n==6 && c.cycle.rng==456);
    pthread_mutex_destroy(&s.model_mu);pthread_mutex_destroy(&s.inference_mu);pthread_cond_destroy(&s.model_cv);
}
int main(void){mixed();bounds_and_ignore();cancelled_pending();unbatched();puts("PASS actual decode executor: mixed batches, sampled RNG, EOS policy, bounds, cancellation, unbatched parity");return 0;}
