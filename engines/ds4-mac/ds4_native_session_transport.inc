/* Mode3 persistent extension of the existing authenticated native transport.
 * Socket/auth/HTTP framing and artifact v2.1 are reused unchanged. All state
 * callbacks run synchronously under the server's existing generation owner. */
#include "ds4_native_session_json.h"
#include "ds4_native_session_negotiate.h"
#include "ds4_native_session_route.h"
#include <pthread.h>
#include <stdatomic.h>
typedef struct {ds4_nf_config http;char tier[9],techniques[4097];uint32_t yoco_keep,local_prefill_max;bool async_decode,catchup_defer;} ds4_ns_config;
typedef struct {
    ds4_native_checkpoint identity;
    float *logits;unsigned char *state;size_t bytes,cursor;
} ds4_ns_anchor;
/* One catch-up delta base->target (<=128 rows): exact copies of every
 * overwritten ring/tail owner at target, the appended rows of the append-only
 * compressed/index owners, the new tokens and the target logits. */
typedef struct ds4_ns_batch {
    struct ds4_ns_batch*next;
    uint32_t context,base,target,span_count,source;
    int32_t*tokens;float*logits;unsigned char*ring,*tail;
    dsv41_c_span spans[55];
    uint64_t ring_offset[55],tail_offset[55],base_bytes[55];
    char tokens_sha256[65],append_sha256[55][65];
    double queued_at;
} ds4_ns_batch;
#define DS4_NS_QUEUE_MAX 64u
#define DS4_NS_STEP_MAX 17u /* ds4_server.c decode loop: int toks[17] */
typedef struct {
    ds4_ns_config*config;char id[33];uint64_t sequence;
    ds4_native_checkpoint synced;bool started,poisoned;
    /* Negotiated at start: min(peer's advertised maximum, DS4_NS_DECODE_DELTA_MAX); 128 for a peer that does not advertise one. */
    uint32_t decode_delta_max,prefill_sweep_max;
    bool include_history_reasoning,force_remote;
    ds4_ns_anchor anchor;
    uint64_t sent_delta_bytes,received_delta_bytes,initial_bytes,decode_batches,prefill_batches;
    double initial_transfer_ms,prefill_delta_ms,decode_delta_ms;
    /* Async catch-up queue. The worker owns sequence/synced while busy; the
     * generation thread reads them only after ds4_ns_q_join. */
    pthread_mutex_t q_mu;pthread_cond_t q_cv;pthread_t q_thread;
    bool q_init,q_thread_started,q_busy,q_failed,q_hold;_Atomic bool q_stop; /* q_hold: defer sends until the reply ends */
    ds4_ns_batch*q_head,*q_tail;uint32_t q_count,batch_source;char q_error[256];
    /* Incremental owner hashes of synced+queued rows (generation thread). */
    bool chain_valid;uint32_t chain_pos;CC_SHA256_CTX chain_tokens,chain_append[8];
    /* One timing line per synchronization (DECODEFIX instrumentation). */
    bool report_pending;uint32_t report_position,report_tokens,report_queued;unsigned report_force,report_async;
    double last_sync_return,report_prev_ms,report_ready_ms,report_work_ms,report_total_ms;
    double verify_rate; /* DECODEFIX2: measured producer owner-verify bytes/s */
    bool remote_accepted; /* DECODEFIX2: Studio answered 202 to the start POST, so remote state exists */
} ds4_ns_session;
static void ds4_ns_anchor_free(ds4_ns_anchor*a){free(a->logits);free(a->state);memset(a,0,sizeof(*a));}
static uint64_t ds4_ns_fingerprint(const char*s){uint64_t v=0;for(unsigned i=0;i<16;i++)v=(v<<4)|(uint64_t)(s[i]<='9'?s[i]-'0':s[i]-'a'+10);return v;}
static int ds4_ns_verify(void*ud,const ds4_native_view*v,char*err,size_t cap){
    ds4_native_checkpoint actual;
    if(!ds4_native_capture(&actual,v,err,cap))return 0;
    if(!ds4_ns_checkpoint_same(ud,&actual)){snprintf(err,cap,"native session actual token/owner checkpoint differs");return 0;}return 1;
}
static bool ds4_ns_open(ds4_ns_session*s,ds4_nf_io*io,const char*method,const char*suffix,
                         const char*media,const char*extra,uint64_t length){
    char h[24576];ds4_ns_text b={h,sizeof(h),true};ds4_nf_config*c=&s->config->http;
    ds4_ns_put(&b,"%s /dsv41-native-offload/sessions/%s%s HTTP/1.1\r\nHost: %s:%u\r\nX-Studio-Token: %s\r\nConnection: close\r\nContent-Length: %llu\r\n",
        method,s->id,suffix,c->host,ntohs(c->broker.sin_port),c->token,(unsigned long long)length);
    if(media)ds4_ns_put(&b,"Content-Type: %s\r\n",media);
    if(extra)ds4_ns_put(&b,"%s",extra);ds4_ns_put(&b,"\r\n");
    if(!b.ok){fprintf(stderr,"native session request header bound failed\n");return false;}
    if(!ds4_nf_connect(c,io)){fprintf(stderr,"native session connect failed host=%s port=%u errno=%d remaining=%.3f cancelled=%d\n",c->host,ntohs(c->broker.sin_port),errno,io->deadline-ds4_nf_now(),io->cancelled?io->cancelled(io->cancel_ud):0);return false;}
    if(!ds4_nf_send(io,h,(size_t)(b.p-h))){fprintf(stderr,"native session header send failed errno=%d\n",errno);return false;}return true;
}
static bool ds4_ns_body(ds4_nf_io*io,int expected,ds4_ns_reply*r){
    int status;uint64_t n;
    if(!ds4_nf_response(io,&status,&n)){fprintf(stderr,"native session HTTP response framing failed idle=%.3f remaining=%.3f errno=%d\n",ds4_nf_now()-io->last_activity,io->deadline-ds4_nf_now(),errno);return false;}
    if(status!=expected||!n||n>DS4_NS_JSON_CAP){fprintf(stderr,"native session HTTP status=%d expected=%d bytes=%llu\n",status,expected,(unsigned long long)n);return false;}
    char*text=malloc((size_t)n+1);if(!text)return false;
    bool ok=ds4_nf_read(io,text,(size_t)n);text[n]=0;
    if(ok)ok=ds4_ns_reply_parse(text,(size_t)n,r);
    if(!ok)fprintf(stderr,"native session JSON response rejected bytes=%llu\n",(unsigned long long)n);
    free(text);return ok;
}
static void ds4_ns_disconnect(ds4_nf_io*io){if(io->fd>=0)close(io->fd);io->fd=-1;}
static bool ds4_ns_identity(ds4_ns_session*s,const ds4_ns_reply*r){
    return !strcmp(s->id,r->session)&&!strcmp(DS4_NF_MODEL_SHA,r->model_sha256);
}
/* See ds4_native_session_negotiate.h (bits 20 / 11: the reply carried the field). */
static bool ds4_ns_negotiate_decode(const ds4_ns_reply*r,uint32_t*effective){
    return ds4_ns_negotiate_decode_values((r->seen&(UINT64_C(1)<<20))!=0,r->maximum_decode_delta_tokens,
        (r->seen&(UINT64_C(1)<<11))!=0,r->maximum_decode_tokens,effective);
}
static bool ds4_ns_poll(ds4_ns_session*s,ds4_nf_io*io,const char*suffix,bool starting,ds4_ns_reply*r){
    for(;;){
        ds4_ns_disconnect(io);
        if(!ds4_ns_open(s,io,"GET",suffix,NULL,NULL,0)||!ds4_ns_body(io,200,r))return false;
        ds4_ns_disconnect(io);
        if(io->progress_session&&r->prefill_total){
            if(strcmp(s->id,r->session)||r->prefill_total!=io->progress_total||r->prefill_processed>r->prefill_total)return false;
            if(!io->progress_reported||r->prefill_processed>io->progress_done){
                ds4_session_report_progress(io->progress_session,"prefill_chunk",(int)r->prefill_processed,(int)r->prefill_total);
                io->progress_done=r->prefill_processed;io->progress_reported=true;
            }
        }
        if(starting){
            if(strcmp(s->id,r->session))return false;
            if(!strcmp(r->state,"ready"))return ds4_ns_identity(s,r)&&r->have_checkpoint&&r->next_sequence==1&&
                r->maximum_prefill_tokens==49152&&ds4_ns_negotiate_decode(r,&s->decode_delta_max)&&r->control_host_bytes==(64u<<20)&&
                ds4_ns_negotiate_yoco_values(s->config->yoco_keep,r->yoco_keep,r->yoco_state_version);
            if(strcmp(r->state,"starting"))return false;
        }else{
            if(r->sequence!=s->sequence)return false;
            if(!strcmp(r->state,"complete"))return ds4_ns_identity(s,r);
            if(strcmp(r->state,"running"))return false;
        }
        /* The broker's independent producer watchdog certifies work progress
         * and changes this state to failure on a stall. A healthy pending
         * response renews transport liveness, never a total-operation cap. */
        ds4_nf_progress_at(io,ds4_nf_now());
        if(ds4_nf_now()>=io->deadline||(io->cancelled&&io->cancelled(io->cancel_ud)))return false;
        (void)poll(NULL,0,50);
    }
}
static void ds4_ns_command_prefix(ds4_ns_text*b,ds4_ns_session*s,const char*op){
    ds4_ns_put(b,"{\"session\":\"%s\",\"model_sha256\":\"%s\",\"sequence\":%llu,\"operation\":\"%s\"",
        s->id,DS4_NF_MODEL_SHA,(unsigned long long)s->sequence,op);
}
static bool ds4_ns_command(ds4_ns_session*s,ds4_nf_io*io,const char*body,size_t n,ds4_ns_reply*r){
    ds4_ns_reply queued;char suffix[80];
    if(!ds4_ns_open(s,io,"POST","/commands","application/json",NULL,n)||!ds4_nf_send(io,body,n)||!ds4_ns_body(io,202,&queued)||
       strcmp(s->id,queued.session)||queued.sequence!=s->sequence)return false;
    snprintf(suffix,sizeof(suffix),"/commands/%llu",(unsigned long long)s->sequence);
    return ds4_ns_poll(s,io,suffix,false,r);
}
static bool ds4_ns_simple(ds4_ns_session*s,ds4_nf_io*io,const char*op,const ds4_native_checkpoint*base,ds4_ns_reply*r){
    char text[8192];ds4_ns_text b={text,sizeof(text),true};ds4_ns_command_prefix(&b,s,op);
    if(base){ds4_ns_put(&b,",\"base\":");ds4_ns_checkpoint_text(&b,base);}ds4_ns_put(&b,"}");
    if(!b.ok||!ds4_ns_command(s,io,text,(size_t)(b.p-text),r))return false;
    s->sequence++;return true;
}
/* ---- Ordered asynchronous RTX catch-up (DECODEFIX) ----------------------
 * The generation thread snapshots each <=128-row delta at a token boundary
 * and returns to decoding. One worker sends the snapshots in order through the
 * unchanged apply_decode_delta protocol. A failure stops the worker, keeps the
 * error and poisons the session at the next boundary, join or shutdown; it is
 * never retried or refilled. */
static void ds4_ns_batch_free(ds4_ns_batch*b){if(!b)return;free(b->tokens);free(b->logits);free(b->ring);free(b->tail);free(b);}
static bool ds4_ns_q_cancelled(void*ud){ds4_ns_session*s=ud;return atomic_load(&s->q_stop);}
static bool ds4_ns_send_batch(ds4_ns_session*s,ds4_ns_batch*b,char*err,size_t cap);
static void*ds4_ns_q_worker(void*ud){
    ds4_ns_session*s=ud;pthread_mutex_lock(&s->q_mu);
    for(;;){
        while(!atomic_load(&s->q_stop)&&(!s->q_head||s->q_failed||s->q_hold))pthread_cond_wait(&s->q_cv,&s->q_mu);
        if(atomic_load(&s->q_stop))break;
        ds4_ns_batch*b=s->q_head;s->q_head=b->next;if(!s->q_head)s->q_tail=NULL;s->q_count--;s->q_busy=true;
        pthread_mutex_unlock(&s->q_mu);
        char err[256]={0};const bool ok=ds4_ns_send_batch(s,b,err,sizeof(err));
        const uint32_t base=b->base,target=b->target;ds4_ns_batch_free(b);
        pthread_mutex_lock(&s->q_mu);s->q_busy=false;
        if(!ok){
            s->q_failed=true;snprintf(s->q_error,sizeof(s->q_error),"async catch-up %u..%u failed: %s",base,target,err[0]?err:"transport");
            for(ds4_ns_batch*n=s->q_head;n;){ds4_ns_batch*x=n->next;ds4_ns_batch_free(n);n=x;}
            s->q_head=s->q_tail=NULL;s->q_count=0;
            fprintf(stderr,"native session failure phase=async catch-up detail=%s\n",s->q_error);
        }
        pthread_cond_broadcast(&s->q_cv);
    }
    pthread_mutex_unlock(&s->q_mu);return NULL;
}
static bool ds4_ns_q_start(ds4_ns_session*s){
    if(!s->q_init){
        if(pthread_mutex_init(&s->q_mu,NULL))return false;
        if(pthread_cond_init(&s->q_cv,NULL)){pthread_mutex_destroy(&s->q_mu);return false;}
        s->q_init=true;s->q_failed=false;s->q_busy=false;atomic_store(&s->q_stop,false);s->q_head=s->q_tail=NULL;s->q_count=0;
    }
    if(!s->q_thread_started){
        if(pthread_create(&s->q_thread,NULL,ds4_ns_q_worker,s))return false;
        s->q_thread_started=true;
    }
    return true;
}
/* Backpressure: blocks only when DS4_NS_QUEUE_MAX snapshots are pending. */
static bool ds4_ns_q_push(ds4_ns_session*s,ds4_ns_batch*b,uint32_t*depth){
    if(!ds4_ns_q_start(s)){ds4_ns_batch_free(b);return false;}
    pthread_mutex_lock(&s->q_mu);
    if(s->q_count>=DS4_NS_QUEUE_MAX&&s->q_hold){s->q_hold=false;pthread_cond_broadcast(&s->q_cv);} /* never deadlock a long deferred reply */
    while(!s->q_failed&&!atomic_load(&s->q_stop)&&s->q_count>=DS4_NS_QUEUE_MAX)pthread_cond_wait(&s->q_cv,&s->q_mu);
    const bool ok=!s->q_failed&&!atomic_load(&s->q_stop);
    if(ok){b->next=NULL;if(s->q_tail)s->q_tail->next=b;else s->q_head=b;s->q_tail=b;s->q_count++;
        if(depth)*depth=s->q_count+(s->q_busy?1u:0u);pthread_cond_broadcast(&s->q_cv);}
    pthread_mutex_unlock(&s->q_mu);
    if(!ok)ds4_ns_batch_free(b);return ok;
}
/* False only after a recorded worker failure. *idle: nothing queued or sending. */
static bool ds4_ns_q_state(ds4_ns_session*s,bool*idle,char*err,size_t cap){
    *idle=true;if(!s->q_init)return true;
    pthread_mutex_lock(&s->q_mu);
    const bool ok=!s->q_failed;*idle=!s->q_head&&!s->q_busy;
    if(!ok&&err&&cap)snprintf(err,cap,"%s",s->q_error);
    pthread_mutex_unlock(&s->q_mu);return ok;
}
/* Strict join: every queued delta is applied and acknowledged by the RTX, or
 * the recorded failure is returned. Required before any other producer use. */
static bool ds4_ns_q_join(ds4_ns_session*s,char*err,size_t cap){
    if(!s->q_init)return true;
    pthread_mutex_lock(&s->q_mu);s->q_hold=false;pthread_cond_broadcast(&s->q_cv); /* a join always releases deferred sends */
    while(!s->q_failed&&!atomic_load(&s->q_stop)&&(s->q_head||s->q_busy))pthread_cond_wait(&s->q_cv,&s->q_mu);
    const bool ok=!s->q_failed&&!s->q_head&&!s->q_busy;
    if(!ok&&err&&cap)snprintf(err,cap,"%s",s->q_failed?s->q_error:"async catch-up stopped");
    pthread_mutex_unlock(&s->q_mu);return ok;
}
/* Cancels an in-flight send (io polls every 100 ms), joins the thread, frees
 * whatever is left. Idempotent; never called from the worker. */
static void ds4_ns_q_stop(ds4_ns_session*s){
    if(!s->q_init)return;
    pthread_mutex_lock(&s->q_mu);atomic_store(&s->q_stop,true);pthread_cond_broadcast(&s->q_cv);pthread_mutex_unlock(&s->q_mu);
    if(s->q_thread_started){pthread_join(s->q_thread,NULL);s->q_thread_started=false;}
    for(ds4_ns_batch*n=s->q_head;n;){ds4_ns_batch*x=n->next;ds4_ns_batch_free(n);n=x;}
    s->q_head=s->q_tail=NULL;s->q_count=0;s->q_busy=false;
    pthread_cond_destroy(&s->q_cv);pthread_mutex_destroy(&s->q_mu);s->q_init=false;atomic_store(&s->q_stop,false);
    s->chain_valid=false;
}
static void ds4_ns_abort(ds4_ns_session*s){
    if(!s)return;s->poisoned=true;
    ds4_ns_q_stop(s);
    ds4_ns_anchor_free(&s->anchor);
    if(s->started){ds4_nf_io io={.fd=-1,.deadline=ds4_nf_now()+3};
        ds4_ns_reply reply;(void)(ds4_ns_open(s,&io,"DELETE","",NULL,NULL,0)&&ds4_ns_body(&io,200,&reply));ds4_ns_disconnect(&io);}
}
static bool ds4_ns_shutdown(ds4_ns_session*s){
    if(!s)return true;
    if(s->started&&!s->poisoned){char err[256]={0};if(!ds4_ns_q_join(s,err,sizeof(err)))s->poisoned=true;}
    ds4_ns_q_stop(s);
    ds4_ns_anchor_free(&s->anchor);
    if(!s->started)return true;
    if(!s->remote_accepted){s->started=false;return true;}
    ds4_nf_io io={.fd=-1,.deadline=ds4_nf_now()+60};ds4_ns_reply r;
    if(!s->poisoned){if(!ds4_ns_simple(s,&io,"close",NULL,&r)||!r.closed)ds4_ns_abort(s);}
    else ds4_ns_abort(s);
    for(;;){
        ds4_ns_disconnect(&io);
        if(!ds4_ns_open(s,&io,"GET","",NULL,NULL,0)||!ds4_ns_body(&io,200,&r)||strcmp(r.session,s->id))break;
        if(r.cleanup_complete){ds4_ns_disconnect(&io);s->started=false;return true;}
        if(ds4_nf_now()>=io.deadline)break;(void)poll(NULL,0,50);
    }
    ds4_ns_disconnect(&io);fprintf(stderr,"native session owned RTX cleanup unconfirmed\n");return false;
}
typedef struct {ds4_nf_io*io;uint64_t remaining;} ds4_ns_stream;
static int ds4_ns_read(void*ud,void*dst,size_t n){
    ds4_ns_stream*s=ud;if(!n||n>65536||n>s->remaining||!ds4_nf_read(s->io,dst,n))return 0;s->remaining-=n;return 1;
}
static int ds4_ns_write(void*ud,void*src,size_t n){
    ds4_ns_stream*s=ud;if(!n||n>65536||n>s->remaining||!ds4_nf_send(s->io,src,n))return 0;s->remaining-=n;return 1;
}
static int ds4_ns_send_page(void*ud,void*src,size_t n){return ds4_nf_send(ud,src,n)?1:0;}
static int ds4_ns_file_read(void*ud,char*dst,int n){
    ds4_ns_stream*s=ud;if(n<=0)return 0;if(!s->remaining)return 0;
    size_t want=(uint64_t)n<s->remaining?(size_t)n:(size_t)s->remaining;if(want>65536)want=65536;
    return ds4_ns_read(ud,dst,want)?(int)want:-1;
}
static bool ds4_ns_import(ds4_ns_session*s,ds4_session*session,ds4_nf_io*io,const ds4_ns_reply*meta,bool initial,char*err,size_t cap){
    const double started=ds4_nf_now();
    int status;uint64_t n;bool ok=false;
    if(!meta->have_target||meta->target.context!=1000001||
       strcmp(meta->kind,initial?"native_initial":"artifact_delta_v2_1")||
       (initial?meta->have_base:(!meta->have_base||!ds4_ns_checkpoint_same(&meta->base,&s->synced))))return false;
    if(!ds4_ns_open(s,io,"GET","/payload",NULL,NULL,0)||!ds4_nf_response(io,&status,&n)||status!=200||n!=meta->bytes)return false;
    ds4_ns_stream stream={io,n};
    if(initial){
        dsv41_c_span spans[55];uint32_t count=0;uint64_t bytes=60+(uint64_t)meta->target.position*4+129280*4;
        if(!dsv41_c_layout(1000001,meta->target.position,spans,&count))goto done;
        for(uint32_t i=0;i<count;i++)bytes+=spans[i].bytes;if(bytes!=n)goto done;
        FILE*f=funopen(&stream,ds4_ns_file_read,NULL,NULL,NULL);if(!f)goto done;
        int rc=ds4_session_load_payload(session,f,n,err,cap);fclose(f);
        ok=!rc&&!stream.remaining&&!ds4_session_native_visit(session,ds4_ns_verify,(void*)&meta->target,err,cap);
    }else{
        int32_t*ids=malloc(49152*4);float*logits=malloc(129280*4);dsv41_c_plan plan;
        if(ids&&logits&&dsv41_c_read_delta_prefix(1000001,s->synced.position,49152,n,ds4_ns_fingerprint(s->id),
             ds4_ns_fingerprint(DS4_NF_MODEL_SHA),ds4_ns_read,&stream,ids,logits,&plan)&&plan.target==meta->target.position)
            ok=!ds4_session_native_apply_delta(session,plan.base,plan.target,ids,logits,ds4_ns_read,&stream,
                ds4_ns_verify,&s->synced,ds4_ns_verify,(void*)&meta->target,err,cap)&&!stream.remaining;
        free(ids);free(logits);
    }
done:
    ds4_ns_disconnect(io);if(!ok)return false;
    ds4_ns_reply ack;
    if(!ds4_ns_simple(s,io,"acknowledge",&meta->target,&ack)||!ack.peer_synchronized||!ack.have_checkpoint||
       !ds4_ns_checkpoint_same(&ack.checkpoint,&meta->target))return false;
    const double elapsed=(ds4_nf_now()-started)*1000;
    s->synced=meta->target;if(initial){s->initial_bytes+=n;s->initial_transfer_ms+=elapsed;}else{s->received_delta_bytes+=n;s->prefill_delta_ms+=elapsed;}
    fprintf(stderr,"NATIVE_SESSION_TRANSFER initial=%u position=%u bytes=%llu milliseconds=%.3f includes=pipe-network-import-hash-ack\n",(unsigned)initial,s->synced.position,(unsigned long long)n,elapsed);
    return true;
}
typedef struct {const ds4_tokens*tokens;ds4_native_checkpoint actual;bool prefix;} ds4_ns_prefix;
static int ds4_ns_prefix_check(void*ud,const ds4_native_view*v,char*err,size_t cap){
    ds4_ns_prefix*p=ud;p->prefix=p->tokens->len>=(int)v->position&&!memcmp(p->tokens->v,v->tokens,(size_t)v->position*4);
    if(!p->prefix){snprintf(err,cap,"conversation prefix differs; explicit new session required");return 0;}
    return ds4_native_capture(&p->actual,v,err,cap);
}
static int ds4_ns_anchor_capture(void*ud,const ds4_native_view*v,char*err,size_t cap){
    ds4_ns_anchor*a=ud;size_t bytes=0;
    for(uint32_t i=0;i<v->span_count;i++)if(v->spans[i].kind!=DSV41_C_COMPRESSED&&v->spans[i].kind!=DSV41_C_INDEX)bytes+=v->spans[i].bytes;
    if(bytes+129280u*4u>24u*1024u*1024u||!ds4_native_capture(&a->identity,v,err,cap))return 0;
    a->state=malloc(bytes);a->logits=malloc(129280u*4u);a->bytes=bytes;
    if(!a->state||!a->logits)return 0;memcpy(a->logits,v->logits,129280u*4u);
    size_t cursor=0;
    for(uint32_t i=0;i<v->span_count;i++){
        if(v->spans[i].kind==DSV41_C_COMPRESSED||v->spans[i].kind==DSV41_C_INDEX)continue;
        for(uint64_t off=0;off<v->spans[i].bytes;){size_t n=(size_t)(v->spans[i].bytes-off);if(n>65536)n=65536;
            if(!v->read_span(v->state,i,off,a->state+cursor,n))return 0;off+=n;cursor+=n;}
    }
    return cursor==bytes;
}
static int ds4_ns_anchor_read(void*ud,void*dst,size_t n){
    ds4_ns_anchor*a=ud;if(n>a->bytes-a->cursor)return 0;memcpy(dst,a->state+a->cursor,n);a->cursor+=n;return 1;
}
static bool ds4_ns_save_rewind(ds4_ns_session*s,ds4_session*session,ds4_nf_io*io,char*err,size_t cap){
    ds4_ns_anchor_free(&s->anchor);ds4_ns_reply r;
    if(ds4_session_native_visit(session,ds4_ns_anchor_capture,&s->anchor,err,cap)||
       !ds4_ns_checkpoint_same(&s->synced,&s->anchor.identity)||
       !ds4_ns_simple(s,io,"save_rewind",&s->synced,&r)||!r.have_checkpoint||!r.peer_synchronized||
       !ds4_ns_checkpoint_same(&s->synced,&r.checkpoint))return false;
    fprintf(stderr,"NATIVE_SESSION_REWIND_ANCHOR position=%u bytes=%zu full_cache_copy=0\n",s->synced.position,s->anchor.bytes+129280u*4u);
    return true;
}
static bool ds4_ns_rewind(ds4_ns_session*s,ds4_session*session,const ds4_tokens*tokens,ds4_nf_io*io,char*err,size_t cap){
    ds4_ns_anchor*a=&s->anchor;const uint32_t target=a->identity.position,old=s->synced.position;char hash[65];
    if(s->include_history_reasoning||!a->state||!target||target>=old||tokens->len<(int)target)return false;
    ds4_native_hash(tokens->v,(size_t)target*4,hash);if(strcmp(hash,a->identity.tokens_sha256))return false;
    if(ds4_session_native_visit(session,ds4_ns_verify,&s->synced,err,cap))return false;
    char text[16384];ds4_ns_text b={text,sizeof(text),true};ds4_ns_reply r;
    ds4_ns_command_prefix(&b,s,"rewind");ds4_ns_put(&b,",\"base\":");ds4_ns_checkpoint_text(&b,&s->synced);
    ds4_ns_put(&b,",\"target\":");ds4_ns_checkpoint_text(&b,&a->identity);ds4_ns_put(&b,"}");
    if(!b.ok||!ds4_ns_command(s,io,text,(size_t)(b.p-text),&r)||!r.have_checkpoint||r.peer_synchronized||
       !ds4_ns_checkpoint_same(&r.checkpoint,&a->identity))return false;s->sequence++;
    a->cursor=0;
    if(ds4_session_native_rewind(session,target,a->logits,ds4_ns_anchor_read,a,
       ds4_ns_verify,&s->synced,ds4_ns_verify,&a->identity,err,cap)||a->cursor!=a->bytes)return false;
    if(!ds4_ns_simple(s,io,"acknowledge",&a->identity,&r)||!r.have_checkpoint||!r.peer_synchronized||
       !ds4_ns_checkpoint_same(&r.checkpoint,&a->identity))return false;
    s->synced=a->identity;
    fprintf(stderr,"NATIVE_SESSION_REWIND base=%u target=%u changed_suffix=%u token_and_all_owner_hashes_verified=1 full_refill=0\n",old,target,(uint32_t)tokens->len-target);
    return true;
}
#include "ds4_native_session_vision.inc"
/* Studio's lazy first-prefill path: the ordinary scheduler handles short
 * prompts until a request actually needs the RTX. No producer is started. */
static bool ds4_ns_local_first(ds4_ns_session*s,ds4_session*session,const ds4_tokens*tokens){
    if(!s||!s->config||s->started||s->poisoned||s->force_remote||!tokens||tokens->len<1)return false;
    const ds4_tokens*live=ds4_session_tokens(session);uint32_t have=0;
    if(live&&live->len>0&&live->len<=tokens->len&&!memcmp(live->v,tokens->v,(size_t)live->len*4))have=(uint32_t)live->len;
    bool local=ds4_ns_local_work(s->config->local_prefill_max,have,(uint32_t)tokens->len,true);
    fprintf(stderr,"NATIVE_SESSION_LOCAL_FIRST route=%s tokens=%d live_prefix=%u mac_work=%u threshold=%u\n",
        local?"mac":"rtx",tokens->len,have,(uint32_t)tokens->len-have,s->config->local_prefill_max);
    return local;
}
static int ds4_ns_decode_sync(ds4_ns_session*,ds4_session*,uint32_t,bool,ds4_nf_cancel_fn,void*,char*,size_t);
/* Ported Studio local append loop. Preserve the negotiated catch-up bound;
 * every Mac-evaluated chunk is acknowledged before another owner uses it. */
static int ds4_ns_local_prefill(ds4_ns_session*s,ds4_session*session,const ds4_tokens*tokens,
                                 const ds4_vision_span*images,size_t image_count,ds4_nf_cancel_fn cancel,void*ud,char*err,size_t cap){
    const double started=ds4_nf_now();const uint32_t base=(uint32_t)ds4_session_pos(session);uint32_t pos=base,chunks=0;int rc=1;
    s->batch_source=1;
    if(ds4_ns_decode_sync(s,session,pos,true,cancel,ud,err,cap))goto done;
    while(pos<(uint32_t)tokens->len){
        if(cancel&&cancel(ud)){ds4_nf_error(err,cap,"local continuation prefill cancelled");goto done;}
        uint32_t next=pos+128u;if(next>(uint32_t)tokens->len)next=(uint32_t)tokens->len;
        ds4_tokens prefix=*tokens;prefix.len=(int)next;
        if((image_count?ds4_session_sync_multimodal(session,&prefix,images,image_count,err,cap):ds4_session_sync(session,&prefix,err,cap))||(uint32_t)ds4_session_pos(session)!=next){
            if(!err[0])ds4_nf_error(err,cap,"local continuation prefill did not reach its chunk");goto done;}
        if(ds4_ns_decode_sync(s,session,next,true,cancel,ud,err,cap))goto done;
        pos=next;chunks++;
    }
    fprintf(stderr,"NATIVE_SESSION_LOCAL_PREFILL base=%u target=%u tokens=%u chunks=%u milliseconds=%.3f threshold=%u catch_up=%s\n",
        base,pos,pos-base,chunks,(ds4_nf_now()-started)*1000,s->config->local_prefill_max,s->config->async_decode?"async":"blocking");
    rc=0;
done:
    s->batch_source=0;return rc;
}
static bool ds4_ns_chain_init(ds4_ns_session*,ds4_session*,char*,size_t);
static bool ds4_ns_start_remote(ds4_ns_session*s,ds4_nf_io*io,const ds4_tokens*first,const ds4_vision_span*images,size_t image_count,ds4_ns_reply*r){
    char hash[65],extra[4608];if(!ds4_nf_token_hash(first,hash)){fprintf(stderr,"native session initial token hash refused count=%d\n",first->len);return false;}
    ds4_ns_text b={extra,sizeof(extra),true};
    ds4_ns_put(&b,"X-DSV41-Model-SHA256: %s\r\nX-DSV41-Tokens-SHA256: %s\r\nX-DSV41-Context: 1000001\r\nX-DSV41-Mode: 3\r\nX-DSV41-Tokens: %d\r\nX-DSV41-Tier: %s\r\n",DS4_NF_MODEL_SHA,hash,first->len,s->config->tier);
    if(*s->config->techniques)ds4_ns_put(&b,"X-DSV41-Techniques: %s\r\n",s->config->techniques);
    if(s->config->yoco_keep)ds4_ns_put(&b,"X-DSV41-Yoco-Keep: %u\r\nX-DSV41-Yoco-State-Version: %u\r\n",s->config->yoco_keep,ds4_ns_yoco_version(s->config->yoco_keep));
    ds4_ns_vision_body wire={0};
    if(image_count){
        if(!ds4_ns_vision_pack(first,images,image_count,&wire))return false;
        ds4_ns_put(&b,"X-DSV41-Body-SHA256: %s\r\n",wire.sha256);
    }
    bool sent=b.ok&&ds4_ns_open(s,io,"POST","",image_count?"application/vnd.studio.dsv41.projected-v1":"application/vnd.studio.dsv41.tokens-le32",extra,image_count?wire.bytes:(uint64_t)first->len*4)&&
        dsv41_c_transfer(ds4_ns_send_page,io,image_count?(void*)wire.data:(void*)first->v,image_count?wire.bytes:(size_t)first->len*4)&&
        ds4_ns_body(io,202,r)&&!strcmp(r->session,s->id);
    free(wire.data);if(!sent)return false;
    s->remote_accepted=true;
    return ds4_ns_poll(s,io,"",true,r)&&r->checkpoint.context==1000001&&r->checkpoint.position==(uint32_t)first->len&&!strcmp(r->checkpoint.tokens_sha256,hash);
}
static int ds4_ns_prefill(ds4_ns_session*s,ds4_session*session,const ds4_tokens*tokens,
                           const ds4_vision_span*images,size_t image_count,
                           ds4_nf_cancel_fn cancel,void*ud,char*err,size_t cap){
    if(!s||s->poisoned||!tokens||tokens->len<1||tokens->len>=1000001)return ds4_nf_error(err,cap,"invalid persistent prefill request");
    ds4_nf_io io={.fd=-1,.deadline=ds4_nf_now()+s->config->http.deadline_seconds,.progress_window=s->config->http.deadline_seconds,.cancelled=cancel,.cancel_ud=ud,
        .progress_session=session,.progress_total=(uint64_t)tokens->len};
    ds4_ns_reply r,meta;bool ok=false;const char*phase="initial identity";
    ds4_session_report_progress(session,"prefill_chunk",s->started?ds4_session_pos(session):0,tokens->len);
    if(s->started){
        const ds4_tokens*now=ds4_session_tokens(session);uint32_t live=now&&now->len>0?(uint32_t)now->len:0;
        bool extends=live&&(uint32_t)tokens->len>=live&&!memcmp(tokens->v,now->v,(size_t)live*4);
        bool images_known=!image_count||ds4_session_vision_prefix_matches(session,images,image_count);
        if(s->include_history_reasoning&&s->anchor.state&&images_known&&ds4_ns_local_work(s->config->local_prefill_max,live,(uint32_t)tokens->len,extends)){
            phase="local continuation prefill";
            if(ds4_ns_local_prefill(s,session,tokens,images,image_count,cancel,ud,err,cap))goto done;
            ok=true;goto done;
        }
        fprintf(stderr,"NATIVE_SESSION_PREFILL_ROUTE route=rtx base=%u target=%d threshold=%u extends=%u\n",live,tokens->len,s->config->local_prefill_max,(unsigned)extends);
        /* A long logical append goes through the same complete fold planner as
         * a cold request. Retire the synchronized owner before opening its
         * replacement; never acknowledge intermediate RTX checkpoints as if
         * the Mac had imported them. Short appends keep the existing reuse path. */
        const uint32_t added=extends?(uint32_t)tokens->len-live:(uint32_t)tokens->len;
        if(added>(s->prefill_sweep_max?s->prefill_sweep_max:49152)){
            phase="retire before full logical replay";
            if(!ds4_ns_shutdown(s)||s->poisoned)goto done;
            memset(&s->synced,0,sizeof(s->synced));s->chain_valid=false;
            fprintf(stderr,"NATIVE_SESSION_PREFILL_REPLAY tokens=%d new_rows=%u final_handoff_only=1\n",tokens->len,added);
        }
    }
    if(!s->started){
        unsigned char nonce[16];if(!random_bytes(nonce,sizeof(nonce)))goto done;
        for(unsigned i=0;i<16;i++)snprintf(s->id+i*2,3,"%02x",nonce[i]);if(!ds4_ns_fingerprint(s->id))goto done;
        s->sequence=1;s->started=true;s->remote_accepted=false;
        ds4_tokens first=*tokens; /* one logical prefill, then one final handoff */
        phase="initial HTTP startup";if(!ds4_ns_start_remote(s,&io,&first,images,image_count,&r))goto done;
        s->prefill_sweep_max=(uint32_t)r.maximum_prefill_sweep_tokens;
        phase="initial export/import";
        if(!ds4_ns_simple(s,&io,"export_initial",&r.checkpoint,&meta)||!ds4_ns_checkpoint_same(&meta.target,&r.checkpoint)||
           !ds4_ns_import(s,session,&io,&meta,true,err,cap))goto done;
        s->prefill_batches++;
        ds4_session_report_progress(session,"prefill_chunk",(int)s->synced.position,tokens->len);
    }
    phase="catch-up join before RTX prefill";
    if(!ds4_ns_q_join(s,err,cap))goto done;
    s->chain_valid=false;
    phase="reasoning-history rewind";
    const ds4_tokens*live=ds4_session_tokens(session);
    if(live&&live->len>0&&(tokens->len<live->len||memcmp(tokens->v,live->v,(size_t)live->len*4))){
        if(!ds4_ns_rewind(s,session,tokens,&io,err,cap))goto done;
    }
    phase="actual prefix verification";ds4_ns_prefix prefix={tokens,{0},false};
    if(ds4_session_native_visit(session,ds4_ns_prefix_check,&prefix,err,cap)||!ds4_ns_checkpoint_same(&prefix.actual,&s->synced))goto done;
    while(s->synced.position<(uint32_t)tokens->len){
        phase="remote checkpoint";if(!ds4_ns_simple(s,&io,"checkpoint",NULL,&r)||!r.have_checkpoint||!r.peer_synchronized||!ds4_ns_checkpoint_same(&r.checkpoint,&s->synced))goto done;
        uint32_t count=(uint32_t)tokens->len-s->synced.position;if(count>49152)count=49152;
        char*text=malloc(1u<<20);if(!text)goto done;ds4_ns_text b={text,1u<<20,true};
        ds4_ns_command_prefix(&b,s,"prefill");ds4_ns_put(&b,",\"base\":");ds4_ns_checkpoint_text(&b,&s->synced);ds4_ns_put(&b,",\"tokens\":[");
        for(uint32_t i=0;i<count;i++)ds4_ns_put(&b,"%s%d",i?",":"",tokens->v[s->synced.position+i]);ds4_ns_put(&b,"]}");
        ds4_tokens chunk={(int32_t*)tokens->v+s->synced.position,(int)count,(int)count};
        phase="incremental prefill";bool sent=image_count?ds4_ns_vision_command(s,&io,&chunk,images,image_count,&r):
            b.ok&&ds4_ns_command(s,&io,text,(size_t)(b.p-text),&r);free(text);
        if(!sent||!r.have_checkpoint||r.peer_synchronized||r.checkpoint.position!=s->synced.position+count)goto done;s->sequence++;
        phase="incremental export/import";if(!ds4_ns_simple(s,&io,"export_delta",&r.checkpoint,&meta)||!ds4_ns_checkpoint_same(&meta.target,&r.checkpoint)||
           !ds4_ns_import(s,session,&io,&meta,false,err,cap))goto done;s->prefill_batches++;
        ds4_session_report_progress(session,"prefill_chunk",(int)s->synced.position,tokens->len);
    }
    phase="projected image identities";
    if(image_count && !ds4_session_restore_vision_identities(session,images,image_count))goto done;
    phase="saved pre-reasoning frontier";
    if((!s->include_history_reasoning||!s->anchor.state)&&!ds4_ns_save_rewind(s,session,&io,err,cap))goto done;
    phase="catch-up hash chain";if(s->config->async_decode&&!ds4_ns_chain_init(s,session,err,cap))goto done;
    ok=true;
done:
    ds4_ns_disconnect(&io);if(!ok){fprintf(stderr,"native session failure phase=%s detail=%s\n",phase,err);ds4_ns_abort(s);ds4_session_invalidate(session);return ds4_nf_error(err,cap,"persistent prefill/hash/transport failed; no refill fallback");}return 0;
}
static void ds4_ns_renew(ds4_nf_io*io,double window){double d=ds4_nf_now()+window;if(io->deadline<d)io->deadline=d;}
/* Hash verification can be silent while the independently guarded producer
 * advances. The work estimate selects a no-progress window, never a total
 * operation deadline; authenticated session status renews that window. */
static uint64_t ds4_ns_owner_bytes(uint32_t position){
    dsv41_c_span sp[55];uint32_t n=0;uint64_t b=0;if(!dsv41_c_layout(1000001,position,sp,&n))return 0;
    for(uint32_t i=0;i<n;i++)b+=sp[i].bytes;return b;
}
static double ds4_ns_reply_window(const ds4_ns_session*s,uint32_t target){
    const double rate=s->verify_rate>0?s->verify_rate:20e6,work=2.0*(double)ds4_ns_owner_bytes(target)/rate;
    return work*4>30?work*4:30;
}
static bool ds4_ns_await_reply(ds4_ns_session*s,ds4_nf_io*io,double window){
    double last=ds4_nf_now(),probe=last;
    for(;;){const double now=ds4_nf_now();
        if(now-last>=window||(io->cancelled&&io->cancelled(io->cancel_ud)))return false;
        struct pollfd p={.fd=io->fd,.events=POLLIN};int r=poll(&p,1,100);
        if(r<0&&errno==EINTR)continue;if(r<0||(r>0&&(p.revents&(POLLERR|POLLNVAL))))return false;
        if(r>0){ds4_nf_progress_at(io,ds4_nf_now());ds4_ns_renew(io,window);return true;}
        if(now>=probe){
            ds4_nf_io status={.fd=-1,.deadline=ds4_nf_now()+3,.cancelled=io->cancelled,.cancel_ud=io->cancel_ud};ds4_ns_reply reply={0};
            bool parsed=ds4_ns_open(s,&status,"GET","",NULL,NULL,0)&&ds4_ns_body(&status,200,&reply);
            bool healthy=parsed&&ds4_ns_identity(s,&reply)&&!strcmp(reply.state,"ready");
            ds4_ns_disconnect(&status);
            if(healthy){last=ds4_nf_now();ds4_ns_renew(io,window);}
            else if(parsed)return false;
            probe=ds4_nf_now()+(window<2?window/4:.5);
        }
    }
}
static void ds4_ns_note_verify(ds4_ns_session*s,uint32_t target,double waited_s){
    if(waited_s>0.05){const double rate=2.0*(double)ds4_ns_owner_bytes(target)/waited_s;
        s->verify_rate=s->verify_rate>0&&s->verify_rate<rate?s->verify_rate:rate;}
}
typedef struct {ds4_ns_session*s;ds4_nf_io*io;ds4_ns_reply reply;double callback_started;} ds4_ns_decode_write;
static int ds4_ns_decode_visit(void*ud,const ds4_native_view*v,char*err,size_t cap){
    const double started=ds4_nf_now();
    ds4_ns_decode_write*w=ud;ds4_ns_session*s=w->s;w->callback_started=started;ds4_native_checkpoint target;dsv41_c_plan plan;
    if(v->context!=s->synced.context||v->position<=s->synced.position||v->position-s->synced.position>(s->decode_delta_max?s->decode_delta_max:DS4_NS_DECODE_DELTA_CHUNK)||
       !dsv41_c_delta(v->context,s->synced.position,v->position,&plan)||!ds4_native_capture(&target,v,err,cap))return 0;
    char prefix_hash[65];ds4_native_hash(v->tokens,(size_t)s->synced.position*4,prefix_hash);
    if(strcmp(prefix_hash,s->synced.tokens_sha256))return 0;
    char extra[16384];ds4_ns_text b={extra,sizeof(extra),true};ds4_ns_put(&b,"X-DSV41-Control: ");
    ds4_ns_command_prefix(&b,s,"apply_decode_delta");ds4_ns_put(&b,",\"base\":");ds4_ns_checkpoint_text(&b,&s->synced);
    ds4_ns_put(&b,",\"target\":");ds4_ns_checkpoint_text(&b,&target);ds4_ns_put(&b,",\"bytes\":%llu}\r\n",(unsigned long long)plan.artifact_bytes);
    if(!b.ok||!ds4_ns_open(s,w->io,"POST","/delta","application/octet-stream",extra,plan.artifact_bytes))return 0;
    ds4_ns_stream stream={w->io,plan.artifact_bytes};
    if(!dsv41_c_write_delta(&plan,ds4_ns_fingerprint(s->id),ds4_ns_fingerprint(DS4_NF_MODEL_SHA),v->tokens+s->synced.position,
       v->logits,v->read_span,v->state,ds4_ns_write,&stream)||stream.remaining||!ds4_ns_body(w->io,200,&w->reply))return 0;
    ds4_ns_reply*r=&w->reply;
    if(!ds4_ns_identity(s,r)||r->sequence!=s->sequence||strcmp(r->status,"complete")||!r->peer_synchronized||r->rtx_decode||
       !(r->seen&(UINT64_C(1)<<15))||r->received_bytes!=plan.artifact_bytes||!r->have_checkpoint||!ds4_ns_checkpoint_same(&r->checkpoint,&target))return 0;
    const double elapsed=(ds4_nf_now()-started)*1000;
    s->sequence++;s->synced=target;s->sent_delta_bytes+=plan.artifact_bytes;s->decode_batches++;s->decode_delta_ms+=elapsed;
    fprintf(stderr,"NATIVE_SESSION_DECODE_DELTA position=%u bytes=%llu milliseconds=%.3f includes=hash-network-apply-ack\n",target.position,(unsigned long long)plan.artifact_bytes,elapsed);return 1;
}
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
/* Append-only owners: compressed/index rows below position/ratio never change
 * after they are written, which is exactly what dsv41_c_delta relies on. */
static int ds4_ns_append_slot(const dsv41_c_span*sp){
    if(sp->index>=4)return -1;
    return sp->kind==DSV41_C_COMPRESSED?(int)sp->index:sp->kind==DSV41_C_INDEX?4+(int)sp->index:-1;
}
static uint64_t ds4_ns_append_bytes(const dsv41_c_span*sp,uint32_t position){
    return (uint64_t)(position/(sp->index<3?2u:1u))*(sp->kind==DSV41_C_COMPRESSED?512u*4u:128u*4u);
}
static void ds4_ns_ctx_hex(const CC_SHA256_CTX*c,char out[65]){CC_SHA256_CTX t=*c;unsigned char d[32];CC_SHA256_Final(d,&t);ds4_native_digest(d,out);}
/* Seed the incremental hashes from the live append-only rows below synced and
 * prove them equal to the acknowledged checkpoint. Once per RTX import/rewind. */
static int ds4_ns_chain_visit(void*ud,const ds4_native_view*v,char*err,size_t cap){
    ds4_ns_session*s=ud;const uint32_t p=s->synced.position;dsv41_c_span at[55];uint32_t n=0;uint8_t page[65536];char hex[65];
    if(v->context!=s->synced.context||v->position<p||!dsv41_c_layout(v->context,p,at,&n)||n!=s->synced.owner_count||
       !dsv41_c_values_valid(v->tokens,p,v->logits)){snprintf(err,cap,"catch-up chain geometry refused");return 0;}
    CC_SHA256_CTX tok;CC_SHA256_Init(&tok);CC_SHA256_Update(&tok,v->tokens,(CC_LONG)((size_t)p*4));ds4_ns_ctx_hex(&tok,hex);
    if(strcmp(hex,s->synced.tokens_sha256)){snprintf(err,cap,"catch-up chain token prefix differs from acknowledged checkpoint");return 0;}
    CC_SHA256_CTX app[8];bool seen[8]={0};
    for(uint32_t j=0;j<n;j++){const int slot=ds4_ns_append_slot(&at[j]);if(slot<0)continue;
        uint32_t i=0;while(i<v->span_count&&(v->spans[i].kind!=at[j].kind||v->spans[i].index!=at[j].index))i++;
        if(i==v->span_count||v->spans[i].bytes<at[j].bytes||seen[slot]){snprintf(err,cap,"catch-up chain owner missing");return 0;}
        CC_SHA256_Init(&app[slot]);seen[slot]=true;
        for(uint64_t off=0;off<at[j].bytes;){size_t m=(size_t)(at[j].bytes-off<sizeof(page)?at[j].bytes-off:sizeof(page));
            if(!v->read_span(v->state,i,off,page,m)){snprintf(err,cap,"catch-up chain owner read failed");return 0;}
            CC_SHA256_Update(&app[slot],page,(CC_LONG)m);off+=m;}
        ds4_ns_ctx_hex(&app[slot],hex);
        if(strcmp(hex,s->synced.owner_sha256[j])){snprintf(err,cap,"catch-up chain owner differs from acknowledged checkpoint");return 0;}
    }
    for(unsigned k=0;k<8;k++)if(!seen[k]){snprintf(err,cap,"catch-up chain owner set incomplete");return 0;}
    s->chain_tokens=tok;memcpy(s->chain_append,app,sizeof(app));s->chain_pos=p;s->chain_valid=true;return 1;
}
typedef struct {ds4_ns_session*s;ds4_ns_batch*b;double callback_started;} ds4_ns_snap;
/* The only work left on the token path: copy <=24 MiB of ring/tail owners plus
 * the appended rows (one GPU-synchronized lease), advance the hash chain. */
static int ds4_ns_snapshot_visit(void*ud,const ds4_native_view*v,char*err,size_t cap){
    ds4_ns_snap*w=ud;ds4_ns_session*s=w->s;w->callback_started=ds4_nf_now();
    const uint32_t base=s->chain_pos,target=v->position;uint64_t ring=0,tail=0;
    if(!s->chain_valid||v->context!=s->synced.context||target<=base||target-base>128||!v->span_count||v->span_count>55||
       !dsv41_c_values_valid(v->tokens+base,target-base,v->logits)){snprintf(err,cap,"catch-up snapshot geometry or values refused");return 0;}
    ds4_ns_batch*b=calloc(1,sizeof(*b));if(!b){snprintf(err,cap,"catch-up snapshot allocation failed");return 0;}
    b->context=v->context;b->base=base;b->target=target;b->span_count=v->span_count;b->source=s->batch_source;
    for(uint32_t i=0;i<v->span_count;i++){b->spans[i]=v->spans[i];
        if(ds4_ns_append_slot(&v->spans[i])<0){b->ring_offset[i]=ring;ring+=v->spans[i].bytes;continue;}
        const uint64_t from=ds4_ns_append_bytes(&v->spans[i],base);
        if(from>v->spans[i].bytes||ds4_ns_append_bytes(&v->spans[i],target)!=v->spans[i].bytes)goto fail;
        b->base_bytes[i]=from;b->tail_offset[i]=tail;tail+=v->spans[i].bytes-from;
    }
    if(ring>24u*1024u*1024u)goto fail;
    b->tokens=malloc((size_t)(target-base)*4);b->logits=malloc(129280u*4u);b->ring=malloc(ring?ring:1);b->tail=malloc(tail?tail:1);
    if(!b->tokens||!b->logits||!b->ring||!b->tail)goto fail;
    memcpy(b->tokens,v->tokens+base,(size_t)(target-base)*4);memcpy(b->logits,v->logits,129280u*4u);
    for(uint32_t i=0;i<v->span_count;i++){const bool append=ds4_ns_append_slot(&v->spans[i])>=0;
        const uint64_t from=append?b->base_bytes[i]:0;unsigned char*dst=append?b->tail+b->tail_offset[i]:b->ring+b->ring_offset[i];
        for(uint64_t off=0;from+off<v->spans[i].bytes;){size_t m=(size_t)(v->spans[i].bytes-from-off);if(m>65536)m=65536;
            if(!v->read_span(v->state,i,from+off,dst+off,m))goto fail;off+=m;}
    }
    CC_SHA256_Update(&s->chain_tokens,b->tokens,(CC_LONG)((size_t)(target-base)*4));ds4_ns_ctx_hex(&s->chain_tokens,b->tokens_sha256);
    for(uint32_t i=0;i<v->span_count;i++){const int slot=ds4_ns_append_slot(&v->spans[i]);if(slot<0)continue;
        const uint64_t m=v->spans[i].bytes-b->base_bytes[i];
        if(m)CC_SHA256_Update(&s->chain_append[slot],b->tail+b->tail_offset[i],(CC_LONG)m);
        ds4_ns_ctx_hex(&s->chain_append[slot],b->append_sha256[i]);
    }
    s->chain_pos=target;w->b=b;return 1;
fail:
    ds4_ns_batch_free(b);if(err&&cap&&!err[0])snprintf(err,cap,"catch-up snapshot owner copy failed");return 0;
}
/* Identical to ds4_native_capture at target: ring/tail owners from the exact
 * copies, append-only owners and tokens from the verified chain. */
static void ds4_ns_batch_checkpoint(const ds4_ns_batch*b,ds4_native_checkpoint*out){
    ds4_native_checkpoint c;memset(&c,0,sizeof(c));
    c.context=b->context;c.position=b->target;c.capture_end=b->target;c.capture_length=b->target<128?b->target:128;c.owner_count=b->span_count;
    memcpy(c.tokens_sha256,b->tokens_sha256,65);ds4_native_hash(b->logits,129280u*4u,c.logits_sha256);
    for(uint32_t i=0;i<b->span_count;i++){
        if(ds4_ns_append_slot(&b->spans[i])>=0)memcpy(c.owner_sha256[i],b->append_sha256[i],65);
        else ds4_native_hash(b->ring+b->ring_offset[i],(size_t)b->spans[i].bytes,c.owner_sha256[i]);
    }
    *out=c;
}
#pragma clang diagnostic pop
static int ds4_ns_batch_span(void*ud,uint32_t index,uint64_t offset,void*dst,size_t n){
    const ds4_ns_batch*b=ud;
    if(!dst||!n||n>65536||index>=b->span_count||offset>b->spans[index].bytes||n>b->spans[index].bytes-offset)return 0;
    if(ds4_ns_append_slot(&b->spans[index])>=0){
        if(offset<b->base_bytes[index])return 0;
        memcpy(dst,b->tail+b->tail_offset[index]+(offset-b->base_bytes[index]),n);
    }else memcpy(dst,b->ring+b->ring_offset[index]+offset,n);
    return 1;
}
/* Worker thread only: the unchanged apply_decode_delta wire exchange and the
 * same reply verification as ds4_ns_decode_visit. */
static bool ds4_ns_send_batch(ds4_ns_session*s,ds4_ns_batch*b,char*err,size_t cap){
    const double started=ds4_nf_now();ds4_native_checkpoint target;dsv41_c_plan plan;ds4_ns_reply r;
    if(b->base!=s->synced.position||b->context!=s->synced.context||!dsv41_c_delta(b->context,b->base,b->target,&plan)||plan.span_count!=b->span_count){
        snprintf(err,cap,"catch-up base differs from acknowledged frontier");return false;}
    for(uint32_t i=0;i<plan.span_count;i++)if(plan.spans[i].kind!=b->spans[i].kind||plan.spans[i].index!=b->spans[i].index||plan.spans[i].bytes!=b->spans[i].bytes){
        snprintf(err,cap,"catch-up owner geometry differs");return false;}
    ds4_ns_batch_checkpoint(b,&target);
    ds4_nf_io io={.fd=-1,.deadline=started+s->config->http.deadline_seconds,.progress_window=s->config->http.deadline_seconds,.cancelled=ds4_ns_q_cancelled,.cancel_ud=s};
    char extra[16384];ds4_ns_text t={extra,sizeof(extra),true};ds4_ns_put(&t,"X-DSV41-Control: ");
    ds4_ns_command_prefix(&t,s,"apply_decode_delta");ds4_ns_put(&t,",\"base\":");ds4_ns_checkpoint_text(&t,&s->synced);
    ds4_ns_put(&t,",\"target\":");ds4_ns_checkpoint_text(&t,&target);ds4_ns_put(&t,",\"bytes\":%llu}\r\n",(unsigned long long)plan.artifact_bytes);
    ds4_ns_stream stream={&io,plan.artifact_bytes};
    bool ok=t.ok&&ds4_ns_open(s,&io,"POST","/delta","application/octet-stream",extra,plan.artifact_bytes)&&
        dsv41_c_write_delta(&plan,ds4_ns_fingerprint(s->id),ds4_ns_fingerprint(DS4_NF_MODEL_SHA),b->tokens,b->logits,
            ds4_ns_batch_span,b,ds4_ns_write,&stream)&&!stream.remaining;
    if(ok){const double sent=ds4_nf_now();ok=ds4_ns_await_reply(s,&io,ds4_ns_reply_window(s,b->target));
        if(ok)ds4_ns_note_verify(s,b->target,ds4_nf_now()-sent);
        else fprintf(stderr,"native session apply_decode_delta reply: no producer progress within %.1f s (work-sized)\n",ds4_ns_reply_window(s,b->target));}
    ok=ok&&ds4_ns_body(&io,200,&r);
    ds4_ns_disconnect(&io);
    ok=ok&&ds4_ns_identity(s,&r)&&r.sequence==s->sequence&&!strcmp(r.status,"complete")&&r.peer_synchronized&&!r.rtx_decode&&
        (r.seen&(UINT64_C(1)<<15))&&r.received_bytes==plan.artifact_bytes&&r.have_checkpoint&&ds4_ns_checkpoint_same(&r.checkpoint,&target);
    if(!ok){snprintf(err,cap,"apply_decode_delta transport, identity or checkpoint refused");return false;}
    const double now=ds4_nf_now(),elapsed=(now-started)*1000;
    s->sequence++;s->synced=target;s->sent_delta_bytes+=plan.artifact_bytes;s->decode_batches++;s->decode_delta_ms+=elapsed;
    fprintf(stderr,"NATIVE_SESSION_DECODE_DELTA position=%u bytes=%llu milliseconds=%.3f queue_ms=%.3f source=%s async=1 includes=hash-network-apply-ack\n",
        target.position,(unsigned long long)plan.artifact_bytes,elapsed,(started-b->queued_at)*1000,b->source?"local-prefill":"decode");
    return true;
}
static bool ds4_ns_chain_init(ds4_ns_session*s,ds4_session*session,char*err,size_t cap){
    s->chain_valid=false;return !ds4_session_native_visit(session,ds4_ns_chain_visit,s,err,cap);
}
static ds4_ns_batch*ds4_ns_snapshot(ds4_ns_session*s,ds4_session*session,double*ready_ms,char*err,size_t cap){
    ds4_ns_snap w={s,NULL,0};const double t=ds4_nf_now();
    const int rc=ds4_session_native_visit(session,ds4_ns_snapshot_visit,&w,err,cap);
    *ready_ms=w.callback_started>0?(w.callback_started-t)*1000:-1;
    if(rc){ds4_ns_batch_free(w.b);return NULL;}return w.b;
}
/* One line per synchronization, printed when the following decode step ends:
 * next_step_ms is the step right after the sync (the unlogged-stall probe). */
static void ds4_ns_sync_report(ds4_ns_session*s,double now){
    if(!s->report_pending)return;s->report_pending=false;
    fprintf(stderr,"NATIVE_SESSION_SYNC_TIMING mode=%s position=%u tokens=%u force=%u prev_step_ms=%.3f ready_ms=%.3f work_ms=%.3f total_ms=%.3f next_step_ms=%.3f queued=%u\n",
        s->report_async?"async":"blocking",s->report_position,s->report_tokens,s->report_force,s->report_prev_ms,s->report_ready_ms,
        s->report_work_ms,s->report_total_ms,(now-s->last_sync_return)*1000,s->report_queued);
}
static int ds4_ns_decode_blocking(ds4_ns_session*s,ds4_session*session,uint32_t position,bool force,
                                   ds4_nf_cancel_fn cancel,void*ud,char*err,size_t cap){
    const uint32_t old=s->synced.position;
    if(position==old)return 0;
    if(position<old||position-old>(s->decode_delta_max?s->decode_delta_max:DS4_NS_DECODE_DELTA_CHUNK))goto bad;
    if(!force&&position-old<32)return 0;
    const double t0=ds4_nf_now();
    ds4_nf_io io={.fd=-1,.deadline=t0+s->config->http.deadline_seconds,.progress_window=s->config->http.deadline_seconds,.cancelled=cancel,.cancel_ud=ud};
    ds4_ns_decode_write w={s,&io,{0},0};
    int rc=ds4_session_native_visit(session,ds4_ns_decode_visit,&w,err,cap);ds4_ns_disconnect(&io);
    if(!rc){const double t1=ds4_nf_now();
        s->report_pending=true;s->report_async=0;s->report_force=force;s->report_position=position;s->report_tokens=position-old;s->report_queued=0;
        s->report_ready_ms=w.callback_started>0?(w.callback_started-t0)*1000:-1;s->report_work_ms=w.callback_started>0?(t1-w.callback_started)*1000:-1;
        return 0;}
bad:ds4_ns_abort(s);ds4_session_invalidate(session);return ds4_nf_error(err,cap,"native decode synchronization failed; session invalidated");
}
/* Snapshot when >=32 rows are pending and the worker is idle, when the next
 * step could pass the 128-row delta bound, or when forced (end of turn, local
 * prefill chunk). Never waits for the network. */
static int ds4_ns_decode_async(ds4_ns_session*s,ds4_session*session,uint32_t position,bool force,char*err,size_t cap){
    bool idle=true;double ready_ms=0;uint32_t depth=0;
    if(!ds4_ns_q_state(s,&idle,err,cap))goto bad;
    if(!s->chain_valid){
        if(!idle){snprintf(err,cap,"catch-up chain invalid with queued deltas");goto bad;}
        if(!ds4_ns_chain_init(s,session,err,cap))goto bad;
    }
    const uint32_t tip=s->chain_pos;
    if(position==tip)return 0;
    if(position<tip||position-tip>128){snprintf(err,cap,"catch-up position %u outside tip %u..+128",position,tip);goto bad;}
    const uint32_t pending=position-tip;
    /* DECODEFIX2 catch-up policy (owner: stream small deltas unless that costs decode,
     * otherwise one clean transfer after the reply). stream: snapshot from 32 rows when the
     * worker is idle and send at once. defer: snapshot only at the 128-row producer bound,
     * hold every send until the reply ends (forced sync) or the next join; with the current
     * 128-row producer limit the end transfer is a burst of <=128-row deltas. */
    const bool defer=s->config->catchup_defer;
    if(!force&&pending+DS4_NS_STEP_MAX<=128&&(defer||pending<32||!idle))return 0;
    const double t0=ds4_nf_now();
    ds4_ns_batch*b=ds4_ns_snapshot(s,session,&ready_ms,err,cap);if(!b)goto bad;
    b->queued_at=ds4_nf_now();
    if(defer&&s->q_init){pthread_mutex_lock(&s->q_mu);s->q_hold=!force;pthread_cond_broadcast(&s->q_cv);pthread_mutex_unlock(&s->q_mu);}
    else if(defer)s->q_hold=!force;
    if(!ds4_ns_q_push(s,b,&depth)){if(ds4_ns_q_state(s,&idle,err,cap))snprintf(err,cap,"catch-up queue refused");goto bad;}
    s->report_pending=true;s->report_async=1;s->report_force=force;s->report_position=position;s->report_tokens=pending;s->report_queued=depth;
    s->report_ready_ms=ready_ms;s->report_work_ms=(ds4_nf_now()-t0)*1000-(ready_ms>0?ready_ms:0);
    return 0;
bad:fprintf(stderr,"native session failure phase=decode catch-up detail=%s\n",err);
    ds4_ns_abort(s);ds4_session_invalidate(session);return ds4_nf_error(err,cap,"native decode synchronization failed; session invalidated");
}
static int ds4_ns_decode_sync(ds4_ns_session*s,ds4_session*session,uint32_t position,bool force,
                               ds4_nf_cancel_fn cancel,void*ud,char*err,size_t cap){
    if(s&&!s->started&&!s->poisoned)return 0; /* DECODEFIX2 lazy: no RTX session yet, nothing to catch up */
    if(!s||!s->started||s->poisoned)return ds4_nf_error(err,cap,"native decode session unavailable");
    const double entered=ds4_nf_now();ds4_ns_sync_report(s,entered);
    const double prev=s->last_sync_return>0?(entered-s->last_sync_return)*1000:-1;
    const int rc=s->config->async_decode?ds4_ns_decode_async(s,session,position,force,err,cap):
        ds4_ns_decode_blocking(s,session,position,force,cancel,ud,err,cap);
    s->last_sync_return=ds4_nf_now();
    if(s->report_pending){s->report_prev_ms=prev;s->report_total_ms=(s->last_sync_return-entered)*1000;}
    return rc;
}
