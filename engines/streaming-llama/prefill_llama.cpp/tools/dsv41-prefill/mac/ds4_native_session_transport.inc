/* Mode3 persistent extension of the existing authenticated native transport.
 * Socket/auth/HTTP framing and artifact v2.1 are reused unchanged. All state
 * callbacks run synchronously under the server's existing generation owner. */
#include "ds4_native_session_json.h"
typedef struct {ds4_nf_config http;char tier[9],techniques[4097];} ds4_ns_config;
typedef struct {
    ds4_native_checkpoint identity;
    float *logits;unsigned char *state;size_t bytes,cursor;
} ds4_ns_anchor;
typedef struct {
    ds4_ns_config*config;char id[33];uint64_t sequence;
    ds4_native_checkpoint synced;bool started,poisoned;
    bool include_history_reasoning;
    ds4_ns_anchor anchor;
    uint64_t sent_delta_bytes,received_delta_bytes,initial_bytes,decode_batches,prefill_batches;
    double initial_transfer_ms,prefill_delta_ms,decode_delta_ms;
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
    if(!ds4_nf_response(io,&status,&n)){fprintf(stderr,"native session HTTP response framing failed\n");return false;}
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
static bool ds4_ns_poll(ds4_ns_session*s,ds4_nf_io*io,const char*suffix,bool starting,ds4_ns_reply*r){
    for(;;){
        ds4_ns_disconnect(io);
        if(!ds4_ns_open(s,io,"GET",suffix,NULL,NULL,0)||!ds4_ns_body(io,200,r))return false;
        ds4_ns_disconnect(io);
        if(starting){
            if(strcmp(s->id,r->session))return false;
            if(!strcmp(r->state,"ready"))return ds4_ns_identity(s,r)&&r->have_checkpoint&&r->next_sequence==1&&
                r->maximum_prefill_tokens==49152&&r->maximum_decode_tokens==49152&&r->control_host_bytes==(64u<<20);
            if(strcmp(r->state,"starting"))return false;
        }else{
            if(r->sequence!=s->sequence)return false;
            if(!strcmp(r->state,"complete"))return ds4_ns_identity(s,r);
            if(strcmp(r->state,"running"))return false;
        }
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
static void ds4_ns_abort(ds4_ns_session*s){
    if(!s)return;s->poisoned=true;
    ds4_ns_anchor_free(&s->anchor);
    if(s->started){ds4_nf_io io={.fd=-1,.deadline=ds4_nf_now()+3};
        ds4_ns_reply reply;(void)(ds4_ns_open(s,&io,"DELETE","",NULL,NULL,0)&&ds4_ns_body(&io,200,&reply));ds4_ns_disconnect(&io);}
}
static bool ds4_ns_shutdown(ds4_ns_session*s){
    if(!s)return true;
    ds4_ns_anchor_free(&s->anchor);
    if(!s->started)return true;
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
static bool ds4_ns_start_remote(ds4_ns_session*s,ds4_nf_io*io,const ds4_tokens*first,ds4_ns_reply*r){
    char hash[65],extra[4608];if(!ds4_nf_token_hash(first,hash)){fprintf(stderr,"native session initial token hash refused count=%d\n",first->len);return false;}
    ds4_ns_text b={extra,sizeof(extra),true};
    ds4_ns_put(&b,"X-DSV41-Model-SHA256: %s\r\nX-DSV41-Tokens-SHA256: %s\r\nX-DSV41-Context: 1000001\r\nX-DSV41-Mode: 3\r\nX-DSV41-Tokens: %d\r\nX-DSV41-Tier: %s\r\n",DS4_NF_MODEL_SHA,hash,first->len,s->config->tier);
    if(*s->config->techniques)ds4_ns_put(&b,"X-DSV41-Techniques: %s\r\n",s->config->techniques);
    if(!b.ok||!ds4_ns_open(s,io,"POST","","application/vnd.studio.dsv41.tokens-le32",extra,(uint64_t)first->len*4)||
       !dsv41_c_transfer(ds4_ns_send_page,io,first->v,(size_t)first->len*4)||!ds4_ns_body(io,202,r)||strcmp(r->session,s->id))return false;
    return ds4_ns_poll(s,io,"",true,r)&&r->checkpoint.context==1000001&&r->checkpoint.position==(uint32_t)first->len&&!strcmp(r->checkpoint.tokens_sha256,hash);
}
static int ds4_ns_prefill(ds4_ns_session*s,ds4_session*session,const ds4_tokens*tokens,
                           ds4_nf_cancel_fn cancel,void*ud,char*err,size_t cap){
    if(!s||s->poisoned||!tokens||tokens->len<1||tokens->len>=1000001)return ds4_nf_error(err,cap,"invalid persistent prefill request");
    ds4_nf_io io={.fd=-1,.deadline=ds4_nf_now()+s->config->http.deadline_seconds,.cancelled=cancel,.cancel_ud=ud};
    ds4_ns_reply r,meta;bool ok=false;const char*phase="initial identity";
    if(!s->started){
        unsigned char nonce[16];if(!random_bytes(nonce,sizeof(nonce)))goto done;
        for(unsigned i=0;i<16;i++)snprintf(s->id+i*2,3,"%02x",nonce[i]);if(!ds4_ns_fingerprint(s->id))goto done;
        s->sequence=1;s->started=true;
        ds4_tokens first=*tokens;if(first.len>49152)first.len=49152;
        phase="initial HTTP startup";if(!ds4_ns_start_remote(s,&io,&first,&r))goto done;
        phase="initial export/import";
        if(!ds4_ns_simple(s,&io,"export_initial",&r.checkpoint,&meta)||!ds4_ns_checkpoint_same(&meta.target,&r.checkpoint)||
           !ds4_ns_import(s,session,&io,&meta,true,err,cap))goto done;
        s->prefill_batches++;
    }
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
        phase="incremental prefill";bool sent=b.ok&&ds4_ns_command(s,&io,text,(size_t)(b.p-text),&r);free(text);
        if(!sent||!r.have_checkpoint||r.peer_synchronized||r.checkpoint.position!=s->synced.position+count)goto done;s->sequence++;
        phase="incremental export/import";if(!ds4_ns_simple(s,&io,"export_delta",&r.checkpoint,&meta)||!ds4_ns_checkpoint_same(&meta.target,&r.checkpoint)||
           !ds4_ns_import(s,session,&io,&meta,false,err,cap))goto done;s->prefill_batches++;
    }
    phase="saved pre-reasoning frontier";
    if((!s->include_history_reasoning||!s->anchor.state)&&!ds4_ns_save_rewind(s,session,&io,err,cap))goto done;
    ok=true;
done:
    ds4_ns_disconnect(&io);if(!ok){fprintf(stderr,"native session failure phase=%s detail=%s\n",phase,err);ds4_ns_abort(s);ds4_session_invalidate(session);return ds4_nf_error(err,cap,"persistent prefill/hash/transport failed; no refill fallback");}return 0;
}
typedef struct {ds4_ns_session*s;ds4_nf_io*io;ds4_ns_reply reply;} ds4_ns_decode_write;
static int ds4_ns_decode_visit(void*ud,const ds4_native_view*v,char*err,size_t cap){
    const double started=ds4_nf_now();
    ds4_ns_decode_write*w=ud;ds4_ns_session*s=w->s;ds4_native_checkpoint target;dsv41_c_plan plan;
    if(v->context!=s->synced.context||v->position<=s->synced.position||v->position-s->synced.position>128||
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
static int ds4_ns_decode_sync(ds4_ns_session*s,ds4_session*session,uint32_t position,bool force,
                               ds4_nf_cancel_fn cancel,void*ud,char*err,size_t cap){
    if(!s||!s->started||s->poisoned)return ds4_nf_error(err,cap,"native decode session unavailable");
    if(position==s->synced.position)return 0;
    if(position<s->synced.position||position-s->synced.position>128)goto bad;
    if(!force&&position-s->synced.position<32)return 0;
    ds4_nf_io io={.fd=-1,.deadline=ds4_nf_now()+s->config->http.deadline_seconds,.cancelled=cancel,.cancel_ud=ud};
    ds4_ns_decode_write w={s,&io,{0}};
    int rc=ds4_session_native_visit(session,ds4_ns_decode_visit,&w,err,cap);ds4_ns_disconnect(&io);if(!rc)return 0;
bad:ds4_ns_abort(s);ds4_session_invalidate(session);return ds4_nf_error(err,cap,"native decode synchronization failed; session invalidated");
}
