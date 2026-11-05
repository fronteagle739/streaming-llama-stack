/* QFN family adapter for the native-session socket/FILE transport.
 * The resident server owns sampling, chat, tools, vision, cancellation and KV.
 * The peer owns only text prefill. No payload file or Python data pipe exists.
 * A long changed conversation is rebuilt on the RTX; local append retains KV.
 */
typedef struct {
    uint32_t position,mtp_rows;uint64_t revision;
    char epoch[33],tokens_sha256[65];
} ds4_qfn_checkpoint;
typedef struct ds4_qfn_session {
    char id[33];uint64_t sequence,delta_budget;
    ds4_qfn_checkpoint synced;bool started;_Atomic bool poisoned;
    uint64_t initial_bytes,forward_bytes,reverse_bytes;
    char rebuild_reason[192];
    ds4_qfn_anchor *anchor;ds4_qfn_checkpoint anchor_checkpoint;char anchor_id[33];
    uint64_t anchor_bytes,rewinds;
    ds4_ns_session catchup;_Atomic uint64_t buffered_bytes;
    uint32_t captured_position,captured_mtp;uint64_t queued_batches;
    int request_cached;
} ds4_qfn_session;
static ds4_qfn_session *ds4_qfn_active_session;
static int ds4_qfn_join(ds4_qfn_session*,char*,size_t);
static void ds4_qfn_session_forget(ds4_qfn_session *s){ds4_ns_q_stop(&s->catchup);ds4_qfn_anchor_free(s->anchor);if(ds4_qfn_active_session==s)ds4_qfn_active_session=NULL;memset(s,0,sizeof(*s));}
static bool ds4_qfn_checkpoint_parse(const char **p,ds4_qfn_checkpoint *c) {
    memset(c,0,sizeof(*c));json_ws(p);if(*(*p)++!='{')return false;unsigned seen=0;
    for(;;) {
        char key[40];unsigned bit=0;uint64_t value=0;
        if(!ds4_nf_ascii(p,key,sizeof(key)))return false;json_ws(p);if(*(*p)++!=':')return false;
        if(!strcmp(key,"position")){bit=1;if(!ds4_nf_u64(p,&value)||value>UINT32_MAX)return false;c->position=(uint32_t)value;}
        else if(!strcmp(key,"mtp_rows")){bit=2;if(!ds4_nf_u64(p,&value)||value>UINT32_MAX)return false;c->mtp_rows=(uint32_t)value;}
        else if(!strcmp(key,"revision")){bit=4;if(!ds4_nf_u64(p,&c->revision)||!c->revision)return false;}
        else if(!strcmp(key,"epoch")){bit=8;if(!ds4_nf_ascii(p,c->epoch,sizeof(c->epoch))||!ds4_nf_hex(c->epoch,32))return false;}
        else if(!strcmp(key,"tokens_sha256")){bit=16;if(!ds4_nf_ascii(p,c->tokens_sha256,sizeof(c->tokens_sha256))||!ds4_nf_hex(c->tokens_sha256,64))return false;}
        else return false;
        if(seen&bit)return false;seen|=bit;json_ws(p);if(**p=='}'){(*p)++;break;}if(*(*p)++!=',')return false;
    }
    return seen==31&&c->position&&c->mtp_rows<=c->position;
}
static void ds4_qfn_checkpoint_text(ds4_ns_text *b,const ds4_qfn_checkpoint *c) {
    ds4_ns_put(b,"{\"position\":%u,\"mtp_rows\":%u,\"tokens_sha256\":\"%s\",\"epoch\":\"%s\",\"revision\":%llu}",
        c->position,c->mtp_rows,c->tokens_sha256,c->epoch,(unsigned long long)c->revision);
}
static bool ds4_qfn_checkpoint_same(const ds4_qfn_checkpoint *a,const ds4_qfn_checkpoint *b) {
    return a->position==b->position&&a->mtp_rows==b->mtp_rows&&a->revision==b->revision&&
        !strcmp(a->epoch,b->epoch)&&!strcmp(a->tokens_sha256,b->tokens_sha256);
}
static bool ds4_qfn_checkpoint_local(ds4_session *s,const ds4_qfn_checkpoint *c) {
    qfn_native_shape shape;uint32_t mtp;char err[128];
    const ds4_tokens *t=ds4_session_tokens(s);
    if(!t||t->len!=(int)c->position||ds4_qfn_session_shape(s,&shape,&mtp,err,sizeof(err))||mtp!=c->mtp_rows)return false;
    unsigned char bytes[32];char hash[65];CC_SHA256(t->v,(CC_LONG)t->len*4,bytes);ds4_nf_digest(bytes,hash);
    return !strcmp(hash,c->tokens_sha256);
}

typedef struct {
    const char *host,*token_file,*model_sha,*compatibility_sha,*state_listen_host;
    const char *producer_binary_sha,*receiver_model_sha,*renderer_sha,*qualification_file,*qualification_sha,*qualification_logits;
    uint32_t port,vision_port,minimum_tokens,local_append_max,stall_seconds;
    uint64_t payload_budget;
    bool verify_state,test_wide_index,test_owner_readback,local_only,test_local_roundtrip;
    bool async_catchup,catchup_defer;uint32_t catchup_max_tokens;uint64_t catchup_buffer_bytes;
} ds4_qfn_peer_config;
static uint32_t ds4_qfn_test_trace_rows; /* opt-in test capture, zero in production */
static ds4_qfn_peer_config ds4_qfn_peer={.port=8095,.minimum_tokens=11776,
    .local_append_max=4096,.stall_seconds=180,.payload_budget=UINT64_C(16)<<30,
    .async_catchup=true,.catchup_max_tokens=128,.catchup_buffer_bytes=UINT64_C(2)<<30};

typedef struct {
    char event[40],model[65],compatibility[65],encoding[32],sha[65],error[512],token[513],nonce[33],session_id[33],owner_epoch[33],anchor_id[33],producer_binary[65];
    uint64_t bytes,canonical,rows,mtp_rows,context,next_sequence,delta_budget,layer,layers,work_rows;
    ds4_qfn_checkpoint checkpoint;bool have_checkpoint;
    double seconds,export_prepare_seconds,gpu_read_seconds,socket_send_seconds,hash_seconds,export_send_seconds;
    bool ok,verify;
} ds4_qfn_peer_reply;

static bool ds4_qfn_line(ds4_nf_io *io,ds4_qfn_peer_reply *out) {
    char text[65537];size_t n=0;
    do {if(n==sizeof(text)-1||!ds4_nf_read(io,text+n,1))return false;}while(text[n++]!='\n');
    text[n]=0;const char *p=text;json_ws(&p);if(*p++!='{')return false;
    memset(out,0,sizeof(*out));uint32_t seen=0;
    for(;;) {
        char key[80];uint32_t bit=0;bool ok=true;
        if(!ds4_nf_ascii(&p,key,sizeof(key)))return false;json_ws(&p);if(*p++!=':')return false;
        if(!strcmp(key,"ok")){bit=1;ok=json_bool(&p,&out->ok);}
        else if(!strcmp(key,"event")){bit=2;ok=ds4_nf_ascii(&p,out->event,sizeof(out->event));}
        else if(!strcmp(key,"error")){bit=4;ok=ds4_nf_ascii(&p,out->error,sizeof(out->error));}
        else if(!strcmp(key,"token")){bit=524288;ok=ds4_nf_ascii(&p,out->token,sizeof(out->token));}
        else if(!strcmp(key,"nonce")){bit=1048576;ok=ds4_nf_ascii(&p,out->nonce,sizeof(out->nonce));}
        else if(!strcmp(key,"model_sha256")){bit=8;ok=ds4_nf_ascii(&p,out->model,sizeof(out->model));}
        else if(!strcmp(key,"compatibility_sha256")){bit=16;ok=ds4_nf_ascii(&p,out->compatibility,sizeof(out->compatibility));}
        else if(!strcmp(key,"wire_encoding")){bit=32;ok=ds4_nf_ascii(&p,out->encoding,sizeof(out->encoding));}
        else if(!strcmp(key,"inline_sha256")){bit=64;ok=ds4_nf_ascii(&p,out->sha,sizeof(out->sha));}
        else if(!strcmp(key,"bytes")){bit=128;ok=ds4_nf_u64(&p,&out->bytes);}
        else if(!strcmp(key,"canonical_bytes")){bit=256;ok=ds4_nf_u64(&p,&out->canonical);}
        else if(!strcmp(key,"rows")){bit=512;ok=ds4_nf_u64(&p,&out->rows);}
        else if(!strcmp(key,"mtp_rows")){bit=1024;ok=ds4_nf_u64(&p,&out->mtp_rows);}
        else if(!strcmp(key,"context")){bit=2048;ok=ds4_nf_u64(&p,&out->context);}
        else if(!strcmp(key,"next_sequence")){bit=2097152;ok=ds4_nf_u64(&p,&out->next_sequence);}
        else if(!strcmp(key,"delta_budget_bytes")){bit=4194304;ok=ds4_nf_u64(&p,&out->delta_budget);}
        else if(!strcmp(key,"checkpoint")){bit=8388608;json_ws(&p);
            if(!strncmp(p,"null",4)){p+=4;out->have_checkpoint=false;}
            else {ok=ds4_qfn_checkpoint_parse(&p,&out->checkpoint);out->have_checkpoint=ok;}}
        else if(!strcmp(key,"session_id")){bit=16777216;ok=ds4_nf_ascii(&p,out->session_id,sizeof(out->session_id))&&(!out->session_id[0]||ds4_nf_hex(out->session_id,32));}
        else if(!strcmp(key,"owner_epoch")){bit=33554432;ok=ds4_nf_ascii(&p,out->owner_epoch,sizeof(out->owner_epoch))&&(!out->owner_epoch[0]||ds4_nf_hex(out->owner_epoch,32));}
        else if(!strcmp(key,"anchor_id")){bit=67108864;ok=ds4_nf_ascii(&p,out->anchor_id,sizeof(out->anchor_id))&&ds4_nf_hex(out->anchor_id,32);}
        else if(!strcmp(key,"producer_binary_sha256")){bit=134217728;ok=ds4_nf_ascii(&p,out->producer_binary,sizeof(out->producer_binary))&&ds4_nf_hex(out->producer_binary,64);}
        else if(!strcmp(key,"layer")){bit=268435456;ok=ds4_nf_u64(&p,&out->layer);}
        else if(!strcmp(key,"layers")){bit=536870912;ok=ds4_nf_u64(&p,&out->layers);}
        else if(!strcmp(key,"work_rows")){bit=1073741824;ok=ds4_nf_u64(&p,&out->work_rows);}
        else if(!strcmp(key,"verify_state")){bit=4096;ok=json_bool(&p,&out->verify);}
        else if(!strcmp(key,"seconds")){bit=8192;ok=json_number(&p,&out->seconds);}
        else if(!strcmp(key,"export_prepare_seconds")){bit=16384;ok=json_number(&p,&out->export_prepare_seconds);}
        else if(!strcmp(key,"producer_gpu_read_seconds")){bit=32768;ok=json_number(&p,&out->gpu_read_seconds);}
        else if(!strcmp(key,"producer_socket_send_seconds")){bit=65536;ok=json_number(&p,&out->socket_send_seconds);}
        else if(!strcmp(key,"producer_inline_hash_seconds")){bit=131072;ok=json_number(&p,&out->hash_seconds);}
        else if(!strcmp(key,"export_and_send_seconds")){bit=262144;ok=json_number(&p,&out->export_send_seconds);}
        else ok=json_skip_value(&p);
        if(!ok||(bit&&(seen&bit)))return false;seen|=bit;json_ws(&p);
        if(*p=='}'){p++;break;}if(*p++!=',')return false;
    }
    json_ws(&p);return !*p&&(seen&1);
}
typedef struct {
    ds4_ns_stream wire;uint64_t at;uint32_t count;
    uint64_t begin[256],end[256];bool verify;CC_SHA256_CTX digest;
    double hash_seconds,read_seconds;
} ds4_qfn_wire_stream;
static bool ds4_qfn_wire_read(ds4_qfn_wire_stream *s,void *dst,size_t n) {
    const double start=ds4_nf_now();
    if(!ds4_ns_read(&s->wire,dst,n))return false;s->read_seconds+=ds4_nf_now()-start;
    if(s->verify){const double hash=ds4_nf_now();CC_SHA256_Update(&s->digest,dst,(CC_LONG)n);s->hash_seconds+=ds4_nf_now()-hash;}
    return true;
}
static float ds4_qfn_half(uint16_t h) {
    /* Widen the source F16 exactly, including subnormals and signed zero. */
    const uint32_t sign=(uint32_t)(h&0x8000)<<16;uint32_t e=(h>>10)&31,m=h&1023,bits;
    if(e==31)bits=sign|0x7f800000u|(m<<13);
    else if(e)bits=sign|((e+112)<<23)|(m<<13);
    else if(!m)bits=sign;
    else {int power=-14;while(!(m&1024)){m<<=1;power--;}bits=sign|((uint32_t)(power+127)<<23)|((m&1023)<<13);}
    float f;memcpy(&f,&bits,4);return f;
}
static int ds4_qfn_file_read(void *ud,char *dst,int requested) {
    ds4_qfn_wire_stream *s=ud;if(requested<=0)return 0;
    size_t n=(size_t)requested>65536?65536:(size_t)requested;uint32_t i=0;
    while(i<s->count&&s->at>=s->end[i])i++;
    if(i<s->count&&s->at>=s->begin[i]) {
        uint16_t half[16384];uint64_t left=s->end[i]-s->at;
        if(n>left)n=(size_t)left;n&=~(size_t)3;
        if(!n||(s->at-s->begin[i])%4||!ds4_qfn_wire_read(s,half,n/2))return -1;
        for(size_t j=0;j<n/4;j++){const float value=ds4_qfn_half(half[j]);memcpy(dst+4*j,&value,4);}
    } else {
        if(i<s->count&&s->at+n>s->begin[i])n=(size_t)(s->begin[i]-s->at);
        if(n>s->wire.remaining)n=(size_t)s->wire.remaining;
        if(!n)return 0;if(!ds4_qfn_wire_read(s,dst,n))return -1;
    }
    s->at+=n;return (int)n;
}
static bool ds4_qfn_peer_enabled(void){return ds4_qfn_peer.host!=NULL&&!ds4_qfn_peer.local_only;}
static int ds4_qfn_remote_image(ds4_engine *engine,const uint8_t *encoded,size_t bytes,
        ds4_vision_embedding *out,char *err,size_t cap) {
    const ds4_qfn_peer_config *c=&ds4_qfn_peer;
    if(ds4_qfn_active_session&&ds4_qfn_join(ds4_qfn_active_session,err,cap))return -1;
    if(!c->vision_port||!bytes||bytes>(64u<<20))return -1;
    ds4_nf_config peer={0};peer.dirfd=-1;peer.broker.sin_family=AF_INET;
    peer.broker.sin_port=htons((uint16_t)c->vision_port);
    if(inet_pton(AF_INET,c->host,&peer.broker.sin_addr)!=1||!ds4_nf_pair_token(c->token_file,peer.token))return -1;
    ds4_nf_io io={.fd=-1,.deadline=ds4_nf_now()+c->stall_seconds,.progress_window=c->stall_seconds};int rc=-1;
    if(!ds4_nf_connect(&peer,&io))goto done;
    char header[4096];int n=snprintf(header,sizeof(header),
        "POST /vision HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer %s\r\nX-QFN-Model-SHA256: %s\r\nX-QFN-Compatibility-SHA256: %s\r\nX-QFN-Producer-Binary-SHA256: %s\r\nContent-Type: application/octet-stream\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
        peer.token,c->model_sha,c->compatibility_sha,c->producer_binary_sha,bytes);
    if(n<0||(size_t)n>=sizeof(header)||!ds4_nf_send(&io,header,(size_t)n)||!ds4_nf_send(&io,encoded,bytes))goto done;
    size_t at=0;do {if(at==sizeof(header)-1||!ds4_nf_read(&io,header+at,1))goto done;at++;}
        while(at<4||memcmp(header+at-4,"\r\n\r\n",4));header[at]=0;
    if(strncmp(header,"HTTP/1.0 200",12)&&strncmp(header,"HTTP/1.1 200",12))goto done;
    const char *length=strstr(header,"Content-Length: ");uint64_t extent=0;
    if(!length)goto done;length+=16;if(!ds4_nf_u64(&length,&extent))goto done;
    uint32_t words[4],grid[4];uint8_t fingerprint[32];
    if(!ds4_nf_read(&io,words,16)||!ds4_nf_read(&io,fingerprint,32)||!ds4_nf_read(&io,grid,16)||
       words[0]!=0x49564c53||words[1]!=1||!words[2]||words[2]>16383||
       words[3]!=(uint32_t)ds4_engine_embd_dim(engine)||!grid[0]||!grid[1]||
       (uint64_t)grid[0]*grid[1]!=words[2]||!grid[2]||!grid[3]||extent!=64u+(uint64_t)words[2]*words[3]*4u)goto done;
    float *values=malloc((size_t)(extent-64));if(!values)goto done;
    if(!ds4_nf_read(&io,values,(size_t)(extent-64))){free(values);goto done;}
    for(uint64_t i=0;i<(uint64_t)words[2]*words[3];i++)if(!isfinite(values[i])){free(values);goto done;}
    memset(out,0,sizeof(*out));out->data=values;out->token_count=words[2];
    out->grid_height=grid[0];out->grid_width=grid[1];out->width=grid[3];out->height=grid[2];
    out->content_width=grid[1]*32u;out->content_height=grid[0]*32u;memcpy(out->fingerprint,fingerprint,32);rc=0;
done:
    memset(peer.token,0,sizeof(peer.token));ds4_ns_disconnect(&io);
    if(rc&&cap)snprintf(err,cap,"RTX QFN lazy image encoder failed; no Mac encoder fallback was used");return rc;
}
static int ds4_qfn_peer_validate(ds4_engine *engine) {
    if(!ds4_qfn_peer.host)return 0;
    const ds4_qfn_peer_config *c=&ds4_qfn_peer;
    if(!ds4_engine_is_qwen4(engine)||!c->token_file||!c->model_sha||!c->compatibility_sha||
       !ds4_nf_hex(c->model_sha,64)||!ds4_nf_hex(c->compatibility_sha,64)||
       !c->producer_binary_sha||!ds4_nf_hex(c->producer_binary_sha,64)||
       !c->receiver_model_sha||!ds4_nf_hex(c->receiver_model_sha,64)||!c->renderer_sha||!ds4_nf_hex(c->renderer_sha,64)||
       !c->qualification_file||!c->qualification_sha||!ds4_nf_hex(c->qualification_sha,64)||
       !c->port||c->port>65535||!c->stall_seconds||!c->payload_budget||
       c->catchup_max_tokens<DS4_NS_STEP_MAX||c->catchup_max_tokens>49152||
       c->catchup_buffer_bytes<(UINT64_C(128)<<20)||c->catchup_buffer_bytes>c->payload_budget)return -1;
    struct in_addr address;
    if(inet_pton(AF_INET,c->host,&address)!=1)return -1;
    if(c->state_listen_host) {
        if(inet_pton(AF_INET,c->state_listen_host,&address)!=1)return -1;
        const uint32_t ip=ntohl(address.s_addr);
        if((ip>>24)!=10&&(ip>>24)!=127&&(ip>>16)!=0xc0a8&&(ip>>20)!=0xac1)return -1;
    }
    int fd=open(c->qualification_file,O_RDONLY|O_NOFOLLOW|O_CLOEXEC);struct stat st;char sha[65];ds4_nf_fields fields;
    bool valid=fd>=0&&!fstat(fd,&st)&&S_ISREG(st.st_mode)&&st.st_uid==geteuid()&&st.st_nlink==1&&
        !(st.st_mode&0022)&&st.st_size>0&&st.st_size<=DS4_NF_QUALIFICATION_CAP&&!flock(fd,LOCK_SH|LOCK_NB);
    char *body=valid?malloc((size_t)st.st_size+1):NULL;if(valid&&!body)valid=false;
    if(valid){valid=pread(fd,body,(size_t)st.st_size,0)==st.st_size;body[st.st_size]=0;
        if(valid){ds4_nf_hash_bytes(body,(size_t)st.st_size,sha);valid=!strcmp(sha,c->qualification_sha)&&ds4_nf_parse_bounded(body,(size_t)st.st_size,&fields,DS4_NF_QUALIFICATION_CAP);}}
    free(body);if(fd>=0)close(fd);if(!valid)return -1;
    const uint64_t required=(UINT64_C(1)<<0)|(UINT64_C(1)<<2)|(UINT64_C(1)<<3)|(UINT64_C(1)<<6)|(UINT64_C(1)<<7)|
        (UINT64_C(1)<<11)|(UINT64_C(1)<<12)|(((UINT64_C(1)<<21)-1)&~((UINT64_C(1)<<14)-1))|(UINT64_C(1)<<23)|(UINT64_C(1)<<24);
    ds4_native_frontier_model_identity model;
    if(!ds4_nf_have(&fields,required)||strcmp(fields.schema,"qfn-native-session-qualification-v1")||
       (strcmp(fields.state,"candidate")&&strcmp(fields.state,"qualified"))||fields.mode!=3||fields.wire_version!=2||
       strcmp(fields.tokenization,"native-final-token-ids-le32-v1")||strcmp(fields.model_sha256,c->model_sha)||
       strcmp(fields.receiver_model_sha256,c->receiver_model_sha)||strcmp(fields.compatibility_sha256,c->compatibility_sha)||
       strcmp(fields.producer_binary_sha256,c->producer_binary_sha)||strcmp(fields.renderer_sha256,c->renderer_sha)||
       !ds4_nf_hex(fields.producer_source_revision,40)||!ds4_engine_native_frontier_identity(engine,&model)||
       fields.receiver_model_size!=model.size||fields.receiver_model_device!=model.device||
       fields.receiver_model_inode!=model.inode||fields.receiver_model_mtime_ns!=model.mtime_ns)return -1;
    char executable[PATH_MAX];uint32_t extent=sizeof(executable);if(_NSGetExecutablePath(executable,&extent))return -1;
    fd=open(executable,O_RDONLY|O_CLOEXEC);valid=fd>=0&&ds4_nf_hash_file(fd,128ull<<20,sha,NULL)&&!strcmp(sha,fields.receiver_binary_sha256);
    if(fd>=0)close(fd);if(!valid)return -1;
    fprintf(stderr,"QFN_INSTALLATION_PINNED state=%s producer=%s receiver=%s renderer=%s source_revision=%s\n",fields.state,
        fields.producer_binary_sha256,fields.receiver_binary_sha256,fields.renderer_sha256,fields.producer_source_revision);
    return 0;
}
typedef struct {ds4_nf_io *io;bool verify;CC_SHA256_CTX hash;uint64_t sent;} ds4_qfn_upload;
static int ds4_qfn_upload_page(void *ud,const void *data,size_t bytes) {
    ds4_qfn_upload *u=ud;if(!ds4_nf_send(u->io,data,bytes))return 0;
    if(u->verify)CC_SHA256_Update(&u->hash,data,(CC_LONG)bytes);u->sent+=bytes;return 1;
}
static int ds4_qfn_delta_read(void *ud,void *data,size_t bytes) {
    ds4_qfn_wire_stream *s=ud;if(!ds4_qfn_wire_read(s,data,bytes))return 0;s->at+=bytes;return 1;
}
/* Same bounded socket callbacks and generation ownership as ds4_ns_send_batch.
 * A reverse delta carries actual target histories, not replayed token IDs. */
typedef struct {
    ds4_qfn_session *owner;unsigned char *body;uint64_t bytes,cursor;
    uint32_t base,target,base_mtp,target_mtp;char tokens_sha256[65],anchor_id[33];bool anchor;
} ds4_qfn_batch;
static int ds4_qfn_send_snapshot(ds4_qfn_session *s,const ds4_qfn_batch *snapshot,
        ds4_nf_cancel_fn cancelled,void *cancel_ud,char *err,size_t cap) {
    const ds4_qfn_peer_config *c=&ds4_qfn_peer;
    if(s->poisoned||snapshot->base!=s->synced.position||snapshot->base_mtp!=s->synced.mtp_rows)
        return ds4_nf_error(err,cap,"QFN ordered snapshot base differs");
    const uint64_t extent=snapshot->bytes;const uint32_t target=snapshot->target,mtp=snapshot->target_mtp;
    const char *prefix=snapshot->tokens_sha256;
    ds4_nf_config peer={0};peer.dirfd=-1;peer.broker.sin_family=AF_INET;peer.broker.sin_port=htons((uint16_t)c->port);
    inet_pton(AF_INET,c->host,&peer.broker.sin_addr);
    ds4_nf_io io={.fd=-1,.deadline=ds4_nf_now()+c->stall_seconds,.progress_window=c->stall_seconds,.cancelled=cancelled,.cancel_ud=cancel_ud};
    ds4_qfn_peer_reply reply={0};int rc=-1,listener=-1,control=-1;char nonce[33]={0};uint32_t port=0;
    if(!ds4_nf_pair_token(c->token_file,peer.token))goto done;
    if(c->state_listen_host) {
        struct sockaddr_in a={.sin_family=AF_INET};inet_pton(AF_INET,c->state_listen_host,&a.sin_addr);
        listener=socket(AF_INET,SOCK_STREAM,0);if(listener<0||bind(listener,(struct sockaddr*)&a,sizeof(a))||listen(listener,1))goto done;
        socklen_t n=sizeof(a);if(getsockname(listener,(struct sockaddr*)&a,&n))goto done;port=ntohs(a.sin_port);
        unsigned char random[16];if(!random_bytes(random,sizeof(random)))goto done;
        for(unsigned i=0;i<16;i++)snprintf(nonce+2*i,3,"%02x",random[i]);
    }
    if(!ds4_nf_connect(&peer,&io))goto done;
    char text[4096];ds4_ns_text b={text,sizeof(text),true};
    ds4_ns_put(&b,"{\"op\":\"pair\",\"token\":\"%s\",\"model_sha256\":\"%s\",\"compatibility_sha256\":\"%s\"}\n",peer.token,c->model_sha,c->compatibility_sha);
    if(!b.ok||!ds4_nf_send(&io,text,(size_t)(b.p-text))||!ds4_qfn_line(&io,&reply)||!reply.ok)goto done;
    if(!strcmp(reply.event,"ready")&&(!ds4_qfn_line(&io,&reply)||!reply.ok))goto done;
    if(strcmp(reply.event,"paired")||strcmp(reply.model,c->model_sha)||strcmp(reply.compatibility,c->compatibility_sha)||strcmp(reply.producer_binary,c->producer_binary_sha))goto done;
    b=(ds4_ns_text){text,sizeof(text),true};
    ds4_ns_put(&b,"{\"op\":\"apply_decode_delta\",\"session\":\"%s\",\"sequence\":%llu,\"base\":",s->id,(unsigned long long)s->sequence);
    ds4_qfn_checkpoint_text(&b,&s->synced);
    ds4_ns_put(&b,",\"target\":%u,\"target_mtp_rows\":%u,\"tokens_sha256\":\"%s\",\"bytes\":%llu,\"max_bytes\":%llu,\"stall_seconds\":%u,\"verify_state\":%s",
        target,mtp,prefix,(unsigned long long)extent,(unsigned long long)s->delta_budget,c->stall_seconds,c->verify_state?"true":"false");
    if(listener>=0)ds4_ns_put(&b,",\"state_callback\":{\"host\":\"%s\",\"port\":%u,\"nonce\":\"%s\"}",c->state_listen_host,port,nonce);
    ds4_ns_put(&b,"}\n");if(!b.ok||!ds4_nf_send(&io,text,(size_t)(b.p-text)))goto done;
    if(listener>=0) {
        ds4_nf_io waiting=io;waiting.fd=listener;if(!ds4_nf_wait(&waiting,POLLIN))goto done;
        int fd=accept(listener,NULL,NULL);if(fd<0)goto done;int mode=fcntl(fd,F_GETFL,0);
        if(mode<0||fcntl(fd,F_SETFL,mode|O_NONBLOCK)){close(fd);goto done;}control=io.fd;io.fd=fd;close(listener);listener=-1;
        if(!ds4_qfn_line(&io,&reply)||!reply.ok||strcmp(reply.event,"state_paired")||strcmp(reply.nonce,nonce)||
           strcmp(reply.token,peer.token)||strcmp(reply.model,c->model_sha)||strcmp(reply.compatibility,c->compatibility_sha))goto done;
    }
    if(!ds4_qfn_line(&io,&reply)||!reply.ok||strcmp(reply.event,"delta_ready")||reply.bytes!=extent)goto done;
    ds4_qfn_upload upload={.io=&io,.verify=c->verify_state};if(upload.verify)CC_SHA256_Init(&upload.hash);
    uint64_t written=0;const double began=ds4_nf_now();
    for(uint64_t at=0;at<extent;) {
        size_t n=(size_t)(extent-at);if(n>65536)n=65536;
        if(!ds4_qfn_upload_page(&upload,snapshot->body+at,n))goto done;at+=n;written+=n;
    }
    if(written!=extent||upload.sent!=extent||
       !ds4_qfn_line(&io,&reply)||!reply.ok||strcmp(reply.event,"delta_applied")||!reply.have_checkpoint||
       reply.bytes!=extent||reply.next_sequence!=s->sequence+1||strcmp(reply.checkpoint.epoch,s->synced.epoch)||
       reply.checkpoint.revision!=s->synced.revision+1||reply.checkpoint.position!=target||
       reply.checkpoint.mtp_rows!=mtp||strcmp(reply.checkpoint.tokens_sha256,prefix))goto done;
    if(upload.verify){unsigned char d[32];char hash[65];CC_SHA256_Final(d,&upload.hash);ds4_nf_digest(d,hash);if(strcmp(hash,reply.sha))goto done;}
    else if(reply.sha[0])goto done;
    fprintf(stderr,"QFN_REVERSE_DELTA base=%u target=%u bytes=%llu seconds=%.6f verify_state=%u\n",
        s->synced.position,reply.checkpoint.position,(unsigned long long)extent,ds4_nf_now()-began,(unsigned)c->verify_state);
    s->synced=reply.checkpoint;s->sequence=reply.next_sequence;s->reverse_bytes+=extent;rc=0;
done:
    if(listener>=0)close(listener);if(control>=0)close(control);memset(peer.token,0,sizeof(peer.token));ds4_ns_disconnect(&io);
    if(rc){s->poisoned=true;if(!err[0])snprintf(err,cap,"QFN reverse delta failed: %s",reply.error[0]?reply.error:"transport/checkpoint");}return rc;
}
static bool ds4_qfn_control_open(ds4_nf_config *peer,ds4_nf_io *io,ds4_qfn_peer_reply *reply) {
    const ds4_qfn_peer_config *c=&ds4_qfn_peer;memset(peer,0,sizeof(*peer));peer->dirfd=-1;
    peer->broker.sin_family=AF_INET;peer->broker.sin_port=htons((uint16_t)c->port);inet_pton(AF_INET,c->host,&peer->broker.sin_addr);
    if(!ds4_nf_pair_token(c->token_file,peer->token)||!ds4_nf_connect(peer,io))return false;
    char text[1536];int n=snprintf(text,sizeof(text),"{\"op\":\"pair\",\"token\":\"%s\",\"model_sha256\":\"%s\",\"compatibility_sha256\":\"%s\"}\n",peer->token,c->model_sha,c->compatibility_sha);
    if(n<0||(size_t)n>=sizeof(text)||!ds4_nf_send(io,text,(size_t)n)||!ds4_qfn_line(io,reply)||!reply->ok)return false;
    if(!strcmp(reply->event,"ready")&&(!ds4_qfn_line(io,reply)||!reply->ok))return false;
    return !strcmp(reply->event,"paired")&&!strcmp(reply->model,c->model_sha)&&!strcmp(reply->compatibility,c->compatibility_sha)&&!strcmp(reply->producer_binary,c->producer_binary_sha);
}
static bool ds4_qfn_control_send(ds4_nf_io *io,const char *text,ds4_ns_text *b,ds4_qfn_peer_reply *reply) {
    return b->ok&&ds4_nf_send(io,text,(size_t)(b->p-text))&&ds4_qfn_line(io,reply)&&reply->ok;
}
static int ds4_qfn_send_anchor(ds4_qfn_session*,const ds4_qfn_batch*,ds4_nf_cancel_fn,void*,char*,size_t);
static int ds4_qfn_join(ds4_qfn_session *s,char *err,size_t cap) {
    if(!ds4_ns_q_join(&s->catchup,err,cap)){s->poisoned=true;return -1;}
    return s->poisoned?ds4_nf_error(err,cap,"QFN ordered catch-up is poisoned"):0;
}
static void ds4_qfn_abort(ds4_qfn_session *s) {s->poisoned=true;ds4_ns_q_stop(&s->catchup);s->started=false;}
static void ds4_qfn_batch_free(void *payload) {
    ds4_qfn_batch *b=payload;if(!b)return;ds4_qfn_session *s=b->owner;
    free(b->body);atomic_fetch_sub(&s->buffered_bytes,b->bytes);free(b);
}
static bool ds4_qfn_batch_send(void *owner,void *payload,char *err,size_t cap) {
    ds4_qfn_session *s=owner;ds4_qfn_batch *b=payload;
    return !(b->anchor?ds4_qfn_send_anchor(s,b,ds4_ns_q_cancelled,&s->catchup,err,cap):
        ds4_qfn_send_snapshot(s,b,ds4_ns_q_cancelled,&s->catchup,err,cap));
}
static int ds4_qfn_capture_page(void *ud,const void *data,size_t n) {
    ds4_qfn_batch *b=ud;if(n>b->bytes-b->cursor)return 0;memcpy(b->body+b->cursor,data,n);b->cursor+=n;return 1;
}
static int ds4_qfn_reserve(ds4_qfn_session *s,uint64_t bytes,ds4_nf_cancel_fn cancel,void *ud,char *err,size_t cap) {
    const ds4_qfn_peer_config *c=&ds4_qfn_peer;
    if(bytes>c->catchup_buffer_bytes||bytes>SIZE_MAX)return ds4_nf_error(err,cap,"QFN catch-up snapshot exceeds explicit buffer budget");
    if(cancel&&cancel(ud))return ds4_nf_error(err,cap,"QFN catch-up capture cancelled");
    if(c->async_catchup&&!ds4_ns_q_start(&s->catchup))return ds4_nf_error(err,cap,"QFN shared worker start failed");
    bool full=false;
    if(s->catchup.q_init){pthread_mutex_lock(&s->catchup.q_mu);full=s->catchup.q_count>=DS4_NS_QUEUE_MAX;pthread_mutex_unlock(&s->catchup.q_mu);}
    if(full||atomic_load(&s->buffered_bytes)>c->catchup_buffer_bytes-bytes)
        if(ds4_qfn_join(s,err,cap))return -1; /* same strict join releases deferred sends */
    atomic_fetch_add(&s->buffered_bytes,bytes);return 0;
}
static ds4_qfn_batch *ds4_qfn_capture(ds4_qfn_session *s,ds4_session *session,uint32_t base,uint32_t base_mtp,
        ds4_nf_cancel_fn cancel,void *ud,char *err,size_t cap) {
    const double began=ds4_nf_now();
    qfn_native_shape shape;uint32_t mtp;qfn_native_plan plan;
    if(ds4_qfn_session_shape(session,&shape,&mtp,err,cap)||
       !qfn_native_geometry(&shape,base,(uint32_t)ds4_session_pos(session),base_mtp,mtp,1,&plan)||
       plan.bytes>UINT64_MAX-40||plan.bytes+40>s->delta_budget)return NULL;
    const uint64_t bytes=plan.bytes+40;if(ds4_qfn_reserve(s,bytes,cancel,ud,err,cap))return NULL;
    ds4_qfn_batch *b=calloc(1,sizeof(*b));
    if(!b){atomic_fetch_sub(&s->buffered_bytes,bytes);return NULL;}
    b->owner=s;b->bytes=bytes;b->base=base;b->base_mtp=base_mtp;b->target=plan.target;b->target_mtp=mtp;b->body=malloc((size_t)bytes);
    if(!b->body)goto failed;
    unsigned char hash[32];const ds4_tokens *tokens=ds4_session_tokens(session);
    CC_SHA256(tokens->v,(CC_LONG)tokens->len*4,hash);ds4_nf_digest(hash,b->tokens_sha256);
    uint64_t written=0;
    if(ds4_qfn_session_delta_write(session,base,base_mtp,s->delta_budget,ds4_qfn_capture_page,b,&written,err,cap)||written!=bytes||b->cursor!=bytes)goto failed;
    fprintf(stderr,"QFN_CATCHUP_CAPTURE base=%u target=%u bytes=%llu seconds=%.6f\n",base,b->target,(unsigned long long)bytes,ds4_nf_now()-began);
    return b;
failed:ds4_qfn_batch_free(b);if(!err[0])snprintf(err,cap,"QFN immutable owner snapshot failed");return NULL;
}
static int ds4_qfn_enqueue(ds4_qfn_session *s,ds4_qfn_batch *b,bool release,char *err,size_t cap) {
    ds4_ns_batch *job=calloc(1,sizeof(*job));if(!job){ds4_qfn_batch_free(b);return -1;}
    job->base=b->base;job->target=b->target;job->family_owner=s;job->family_payload=b;job->family_free=ds4_qfn_batch_free;job->family_send=ds4_qfn_batch_send;
    uint32_t depth=0;
    if(ds4_qfn_peer.catchup_defer&&s->catchup.q_init){pthread_mutex_lock(&s->catchup.q_mu);s->catchup.q_hold=!release;pthread_cond_broadcast(&s->catchup.q_cv);pthread_mutex_unlock(&s->catchup.q_mu);}
    const bool anchor=b->anchor;const uint32_t base=b->base,target=b->target;
    if(!ds4_ns_q_push(&s->catchup,job,&depth))return ds4_nf_error(err,cap,"QFN shared ordered queue refused snapshot");
    s->queued_batches++;fprintf(stderr,"QFN_CATCHUP_QUEUED kind=%s base=%u target=%u depth=%u buffered_bytes=%llu\n",
        anchor?"anchor":"owners",base,target,depth,(unsigned long long)atomic_load(&s->buffered_bytes));return 0;
}
static int ds4_qfn_decode_capture(ds4_qfn_session *s,ds4_session *session,bool force,bool release,
        ds4_nf_cancel_fn cancel,void *ud,char *err,size_t cap) {
    if(!s->started)return 0;if(s->poisoned)return ds4_nf_error(err,cap,"QFN catch-up owner is poisoned");
    const uint32_t position=(uint32_t)ds4_session_pos(session);const ds4_qfn_peer_config *c=&ds4_qfn_peer;
    if(release&&s->catchup.q_init){pthread_mutex_lock(&s->catchup.q_mu);s->catchup.q_hold=false;pthread_cond_broadcast(&s->catchup.q_cv);pthread_mutex_unlock(&s->catchup.q_mu);}
    if(!c->async_catchup) {
        if(!force)return 0;
        if(ds4_qfn_join(s,err,cap))return -1;
        if(position==s->synced.position)return ds4_qfn_checkpoint_local(session,&s->synced)?0:-1;
        ds4_qfn_batch *b=ds4_qfn_capture(s,session,s->synced.position,s->synced.mtp_rows,cancel,ud,err,cap);if(!b)return -1;
        const int rc=ds4_qfn_send_snapshot(s,b,cancel,ud,err,cap);ds4_qfn_batch_free(b);
        if(!rc){s->captured_position=s->synced.position;s->captured_mtp=s->synced.mtp_rows;}return rc;
    }
    bool idle=true;if(!ds4_ns_q_state(&s->catchup,&idle,err,cap)){s->poisoned=true;return -1;}
    if(position<s->captured_position)return ds4_nf_error(err,cap,"QFN async frontier moved behind queued owners");
    if(position==s->captured_position)return 0;
    if(!force&&position-s->captured_position+DS4_NS_STEP_MAX<=c->catchup_max_tokens)return 0;
    ds4_qfn_batch *b=ds4_qfn_capture(s,session,s->captured_position,s->captured_mtp,cancel,ud,err,cap);if(!b)return -1;
    const uint32_t target=b->target,mtp=b->target_mtp;
    if(ds4_qfn_enqueue(s,b,release,err,cap))return -1;s->captured_position=target;s->captured_mtp=mtp;return 0;
}
/* Strict synchronization is reserved for producer reuse, not reply delivery. */
static int ds4_qfn_decode_sync(ds4_qfn_session *s,ds4_session *session,ds4_nf_cancel_fn cancel,void *ud,char *err,size_t cap) {
    if(ds4_qfn_join(s,err,cap))return -1;
    if(!s->started)return 0;
    const ds4_tokens *tokens=ds4_session_tokens(session);if(!tokens)return -1;
    if(tokens->len<(int)s->synced.position){ds4_qfn_session_forget(s);return 0;}
    unsigned char hash[32];char prefix[65];CC_SHA256(tokens->v,(CC_LONG)s->synced.position*4,hash);ds4_nf_digest(hash,prefix);
    if(strcmp(prefix,s->synced.tokens_sha256)){ds4_qfn_session_forget(s);return 0;}
    if(ds4_qfn_decode_capture(s,session,true,true,cancel,ud,err,cap))return -1;
    return ds4_qfn_join(s,err,cap);
}
static int ds4_qfn_send_anchor(ds4_qfn_session *s,const ds4_qfn_batch *snapshot,
        ds4_nf_cancel_fn cancel,void *ud,char *err,size_t cap) {
    const ds4_qfn_peer_config *c=&ds4_qfn_peer;const double began=ds4_nf_now();
    const char *anchor=snapshot->anchor_id;
    if(s->synced.position!=snapshot->target||s->synced.mtp_rows!=snapshot->target_mtp||
       strcmp(s->synced.tokens_sha256,snapshot->tokens_sha256))return ds4_nf_error(err,cap,"queued anchor target differs");
    ds4_nf_config peer;ds4_nf_io io={.fd=-1,.deadline=ds4_nf_now()+c->stall_seconds,.progress_window=c->stall_seconds,.cancelled=cancel,.cancel_ud=ud};
    ds4_qfn_peer_reply reply={0};int rc=-1;
    if(!ds4_qfn_control_open(&peer,&io,&reply))goto done;
    if(strcmp(reply.session_id,s->id)||strcmp(reply.owner_epoch,s->synced.epoch))goto done;
    char text[2048];ds4_ns_text b={text,sizeof(text),true};
    ds4_ns_put(&b,"{\"op\":\"save_rewind\",\"session\":\"%s\",\"sequence\":%llu,\"anchor_id\":\"%s\",\"base\":",s->id,(unsigned long long)s->sequence,anchor);
    ds4_qfn_checkpoint_text(&b,&s->synced);ds4_ns_put(&b,"}\n");
    if(!ds4_qfn_control_send(&io,text,&b,&reply)||strcmp(reply.event,"anchor_saved")||strcmp(reply.anchor_id,anchor)||
       !reply.have_checkpoint||!ds4_qfn_checkpoint_same(&s->synced,&reply.checkpoint)||reply.next_sequence!=s->sequence+1)goto done;
    s->sequence=reply.next_sequence;s->anchor_checkpoint=s->synced;memcpy(s->anchor_id,anchor,33);rc=0;
    fprintf(stderr,"QFN_PAIRED_ANCHOR position=%u mutable_bytes=%llu seconds=%.6f full_cache_copy=0\n",s->synced.position,(unsigned long long)s->anchor_bytes,ds4_nf_now()-began);
done:
    memset(peer.token,0,sizeof(peer.token));ds4_ns_disconnect(&io);
    if(rc){s->poisoned=true;if(!err[0])snprintf(err,cap,"QFN paired anchor save failed: %s",reply.error[0]?reply.error:"epoch/checkpoint/transport");}return rc;
}
static int ds4_qfn_save_rewind(ds4_qfn_session *s,ds4_session *session,
        ds4_nf_cancel_fn cancel,void *ud,char *err,size_t cap) {
    /* Finish the previous anchor before replacing its generation-owned copy. */
    if(ds4_qfn_join(s,err,cap)||!s->started)return -1;
    if(ds4_qfn_decode_capture(s,session,true,false,cancel,ud,err,cap))return -1;
    if(ds4_qfn_anchor_save(session,&s->anchor,s->delta_budget,&s->anchor_bytes,err,cap))return -1;
    qfn_native_shape shape;uint32_t mtp;
    if(ds4_qfn_session_shape(session,&shape,&mtp,err,cap)||ds4_qfn_reserve(s,0,cancel,ud,err,cap))return -1;
    ds4_qfn_batch *b=calloc(1,sizeof(*b));if(!b)return ds4_nf_error(err,cap,"QFN anchor allocation failed");
    b->owner=s;b->anchor=true;b->base=b->target=(uint32_t)ds4_session_pos(session);b->base_mtp=b->target_mtp=mtp;
    unsigned char random[16],hash[32];if(!random_bytes(random,sizeof(random))){ds4_qfn_batch_free(b);return -1;}
    for(unsigned i=0;i<16;i++)snprintf(b->anchor_id+2*i,3,"%02x",random[i]);
    const ds4_tokens *tokens=ds4_session_tokens(session);CC_SHA256(tokens->v,(CC_LONG)tokens->len*4,hash);ds4_nf_digest(hash,b->tokens_sha256);
    ds4_qfn_active_session=s;
    if(ds4_qfn_peer.async_catchup)return ds4_qfn_enqueue(s,b,false,err,cap);
    const int rc=ds4_qfn_send_anchor(s,b,cancel,ud,err,cap);ds4_qfn_batch_free(b);return rc;
}
/* Returns 1 only when the producer epoch was lost and a full rebuild is
 * necessary; the caller preserves and visibly reports that reason. */
static int ds4_qfn_rewind(ds4_qfn_session *s,ds4_session *session,ds4_nf_cancel_fn cancel,void *ud,char *err,size_t cap) {
    if(ds4_qfn_join(s,err,cap))return -1;
    if(!s->started||!s->anchor||!s->anchor_id[0])return ds4_nf_error(err,cap,"QFN paired rewind has no saved anchor");
    const ds4_qfn_peer_config *c=&ds4_qfn_peer;ds4_nf_config peer;
    ds4_nf_io io={.fd=-1,.deadline=ds4_nf_now()+c->stall_seconds,.progress_window=c->stall_seconds,.cancelled=cancel,.cancel_ud=ud};
    ds4_qfn_peer_reply reply={0};int rc=-1;const uint32_t old=s->synced.position;const double began=ds4_nf_now();
    if(!ds4_qfn_control_open(&peer,&io,&reply))goto done;
    if(strcmp(reply.session_id,s->id)||strcmp(reply.owner_epoch,s->synced.epoch)) {
        s->started=false;s->poisoned=false;snprintf(s->rebuild_reason,sizeof(s->rebuild_reason),"The RTX owner epoch changed before rewind; rebuilding its lost checkpoint.");rc=1;goto done;
    }
    char text[3072];ds4_ns_text b={text,sizeof(text),true};
    ds4_ns_put(&b,"{\"op\":\"rewind\",\"session\":\"%s\",\"sequence\":%llu,\"anchor_id\":\"%s\",\"base\":",s->id,(unsigned long long)s->sequence,s->anchor_id);
    ds4_qfn_checkpoint_text(&b,&s->synced);ds4_ns_put(&b,",\"anchor\":");ds4_qfn_checkpoint_text(&b,&s->anchor_checkpoint);
    ds4_ns_put(&b,",\"test_owner_readback\":%s}\n",c->test_owner_readback?"true":"false");
    if(!ds4_qfn_control_send(&io,text,&b,&reply)||strcmp(reply.event,"rewound")||strcmp(reply.anchor_id,s->anchor_id)||
       !reply.have_checkpoint||reply.next_sequence!=s->sequence+1||reply.checkpoint.position!=s->anchor_checkpoint.position||
       reply.checkpoint.mtp_rows!=s->anchor_checkpoint.mtp_rows||strcmp(reply.checkpoint.epoch,s->synced.epoch)||
       reply.checkpoint.revision!=s->synced.revision+1||strcmp(reply.checkpoint.tokens_sha256,s->anchor_checkpoint.tokens_sha256))goto done;
    const ds4_qfn_checkpoint target=reply.checkpoint;
    if(ds4_qfn_anchor_restore(session,s->anchor,err,cap)||!ds4_qfn_checkpoint_local(session,&target))goto done;
    if(c->test_owner_readback) {
        uint64_t checked=0;if(ds4_qfn_anchor_check(session,s->anchor,&checked,err,cap))goto done;
        fprintf(stderr,"QFN_TEST_ANCHOR_READBACK bytes=%llu position=%u actual_native_owners=1 matched=1\n",(unsigned long long)checked,target.position);
    }
    b=(ds4_ns_text){text,sizeof(text),true};
    ds4_ns_put(&b,"{\"op\":\"acknowledge\",\"session\":\"%s\",\"sequence\":%llu,\"checkpoint\":",s->id,(unsigned long long)reply.next_sequence);
    ds4_qfn_checkpoint_text(&b,&target);ds4_ns_put(&b,"}\n");
    if(!ds4_qfn_control_send(&io,text,&b,&reply)||strcmp(reply.event,"acknowledged")||!reply.have_checkpoint||
       !ds4_qfn_checkpoint_same(&target,&reply.checkpoint)||reply.next_sequence!=s->sequence+2)goto done;
    s->synced=target;s->sequence=reply.next_sequence;s->captured_position=target.position;s->captured_mtp=target.mtp_rows;s->rewinds++;rc=0;
    fprintf(stderr,"QFN_PAIRED_REWIND base=%u target=%u seconds=%.6f full_refill=0 acknowledged=1\n",old,target.position,ds4_nf_now()-began);
done:
    memset(peer.token,0,sizeof(peer.token));ds4_ns_disconnect(&io);
    if(rc<0){s->poisoned=true;if(!err[0])snprintf(err,cap,"QFN paired rewind failed: %s",reply.error[0]?reply.error:"epoch/checkpoint/transport");}return rc;
}
/* Test inspection exercises the same explicit checkpoint command as DSV;
 * production handover already validates its base against actual source state. */
static int ds4_qfn_checkpoint_probe(ds4_qfn_session *s,char *err,size_t cap) {
    if(!s->started)return 0;if(ds4_qfn_join(s,err,cap))return -1;
    ds4_nf_config peer;const ds4_qfn_peer_config *c=&ds4_qfn_peer;
    ds4_nf_io io={.fd=-1,.deadline=ds4_nf_now()+c->stall_seconds,.progress_window=c->stall_seconds};
    ds4_qfn_peer_reply reply={0};int rc=-1;
    if(!ds4_qfn_control_open(&peer,&io,&reply)||strcmp(reply.session_id,s->id)||strcmp(reply.owner_epoch,s->synced.epoch))goto done;
    char text[512];ds4_ns_text b={text,sizeof(text),true};
    ds4_ns_put(&b,"{\"op\":\"checkpoint\",\"session\":\"%s\",\"sequence\":%llu}\n",s->id,(unsigned long long)s->sequence);
    if(!ds4_qfn_control_send(&io,text,&b,&reply)||strcmp(reply.event,"checkpoint")||!reply.have_checkpoint||
       !ds4_qfn_checkpoint_same(&s->synced,&reply.checkpoint)||reply.next_sequence!=s->sequence+1)goto done;
    s->sequence=reply.next_sequence;rc=0;
done:memset(peer.token,0,sizeof(peer.token));ds4_ns_disconnect(&io);
    if(rc){s->poisoned=true;if(!err[0])snprintf(err,cap,"QFN actual source checkpoint differs: %s",reply.error[0]?reply.error:"transport/epoch/frontier");}return rc;
}
static int ds4_qfn_peer_prefill(ds4_engine *engine,ds4_session *session,ds4_qfn_session *persistent,const ds4_tokens *prompt,
    const ds4_vision_span *images,size_t image_count,int context,ds4_nf_cancel_fn cancelled,void *cancel_ud,char *err,size_t cap) {
    const ds4_qfn_peer_config *c=&ds4_qfn_peer;
    int live=ds4_session_pos(session),common=ds4_session_common_prefix(session,prompt);
    if(ds4_qfn_join(persistent,err,cap))return -1;
    ds4_qfn_active_session=persistent;
    if(persistent->started&&persistent->anchor&&common<live&&common>=(int)persistent->anchor_checkpoint.position) {
        if(ds4_qfn_decode_sync(persistent,session,cancelled,cancel_ud,err,cap))return -1;
        int rewind=ds4_qfn_rewind(persistent,session,cancelled,cancel_ud,err,cap);if(rewind<0)return -1;
        if(!rewind){live=ds4_session_pos(session);common=ds4_session_common_prefix(session,prompt);
            if(common==live&&prompt->len==live){persistent->request_cached=live;return 0;}}
    }
    bool delta=persistent->started&&common==live&&live>0&&prompt->len>live;
    if(delta&&ds4_qfn_decode_sync(persistent,session,cancelled,cancel_ud,err,cap))return -1;
    delta=delta&&persistent->started;
    if(!delta) {
        char reason[192];memcpy(reason,persistent->rebuild_reason,sizeof(reason));
        ds4_qfn_session_forget(persistent);memcpy(persistent->rebuild_reason,reason,sizeof(reason));unsigned char random[16];
        if(!random_bytes(random,sizeof(random)))return ds4_nf_error(err,cap,"QFN session identity creation failed");
        for(unsigned i=0;i<16;i++)snprintf(persistent->id+2*i,3,"%02x",random[i]);persistent->sequence=1;
    }
    ds4_nf_config peer={0};peer.dirfd=-1;peer.broker.sin_family=AF_INET;
    peer.broker.sin_port=htons((uint16_t)c->port);inet_pton(AF_INET,c->host,&peer.broker.sin_addr);
    if(!ds4_nf_pair_token(c->token_file,peer.token))return ds4_nf_error(err,cap,"QFN pairing token cannot be read");
    ds4_nf_io io={.fd=-1,.deadline=ds4_nf_now()+c->stall_seconds,.progress_window=c->stall_seconds,
        .cancelled=cancelled,.cancel_ud=cancel_ud};
    const double request_start=ds4_nf_now();double paired_at=0,sent_at=0,prefilled_at=0,producer_seconds=0;
    int rc=-1,listener=-1,control=-1;bool mutated=false;ds4_qfn_peer_reply reply={0};const char *phase="connect";
    char nonce[33]={0};uint32_t callback_port=0;
    if(c->state_listen_host) {
        struct sockaddr_in address={.sin_family=AF_INET};
        if(inet_pton(AF_INET,c->state_listen_host,&address.sin_addr)!=1)goto done;
        listener=socket(AF_INET,SOCK_STREAM,0);if(listener<0)goto done;
        if(bind(listener,(struct sockaddr*)&address,sizeof(address))||listen(listener,1))goto done;
        socklen_t length=sizeof(address);if(getsockname(listener,(struct sockaddr*)&address,&length))goto done;
        callback_port=ntohs(address.sin_port);
        unsigned char bytes[16];if(!random_bytes(bytes,sizeof(bytes)))goto done;
        for(unsigned i=0;i<16;i++)snprintf(nonce+2*i,3,"%02x",bytes[i]);
    }
    if(!ds4_nf_connect(&peer,&io))goto done;
    char header[1600];int n=snprintf(header,sizeof(header),
        "{\"op\":\"pair\",\"token\":\"%s\",\"model_sha256\":\"%s\",\"compatibility_sha256\":\"%s\"}\n",
        peer.token,c->model_sha,c->compatibility_sha);
    if(n<0||(size_t)n>=sizeof(header)||!ds4_nf_send(&io,header,(size_t)n))goto done;
    /* The first connection also receives the existing ready event. */
    if(!ds4_qfn_line(&io,&reply)||!reply.ok)goto done;
    if(!strcmp(reply.event,"ready")&&(!ds4_qfn_line(&io,&reply)||!reply.ok))goto done;
    if(strcmp(reply.event,"paired")||strcmp(reply.model,c->model_sha)||
       strcmp(reply.compatibility,c->compatibility_sha)||strcmp(reply.producer_binary,c->producer_binary_sha)||reply.context<(uint32_t)prompt->len||!reply.delta_budget)goto done;
    persistent->delta_budget=reply.delta_budget<c->payload_budget?reply.delta_budget:c->payload_budget;
    if(delta&&(strcmp(reply.session_id,persistent->id)||strcmp(reply.owner_epoch,persistent->synced.epoch))) {
        fprintf(stderr,"QFN_PEER_REBUILD reason=producer owner epoch changed previous_position=%u\n",persistent->synced.position);
        snprintf(persistent->rebuild_reason,sizeof(persistent->rebuild_reason),"The RTX producer restarted (including the lazy image lifecycle); rebuilding its lost checkpoint before decoding.");
        persistent->started=false;persistent->sequence=1;persistent->poisoned=false;delta=false;
        unsigned char random[16];if(!random_bytes(random,sizeof(random)))goto done;
        for(unsigned i=0;i<16;i++)snprintf(persistent->id+2*i,3,"%02x",random[i]);
    }
    paired_at=ds4_nf_now();
    n=snprintf(header,sizeof(header),"{\"op\":\"handover\",\"wire_encoding\":\"%s\",\"verify_state\":%s,\"max_bytes\":%llu,",
        c->test_wide_index?"f32-index-v1":"f16-index-v1",c->verify_state?"true":"false",(unsigned long long)c->payload_budget);
    if(n<0||(size_t)n>=sizeof(header)||!ds4_nf_send(&io,header,(size_t)n))goto done;
    n=snprintf(header,sizeof(header),"\"session\":\"%s\",\"sequence\":%llu,",persistent->id,(unsigned long long)persistent->sequence);
    if(n<0||(size_t)n>=sizeof(header)||!ds4_nf_send(&io,header,(size_t)n))goto done;
    if(delta) {
        ds4_ns_text b={header,sizeof(header),true};ds4_ns_put(&b,"\"base\":");ds4_qfn_checkpoint_text(&b,&persistent->synced);ds4_ns_put(&b,",");
        if(!b.ok||!ds4_nf_send(&io,header,(size_t)(b.p-header)))goto done;
    }
    if(listener>=0) {
        n=snprintf(header,sizeof(header),"\"state_callback\":{\"host\":\"%s\",\"port\":%u,\"nonce\":\"%s\"},",c->state_listen_host,callback_port,nonce);
        if(n<0||(size_t)n>=sizeof(header)||!ds4_nf_send(&io,header,(size_t)n))goto done;
    }
    if(!ds4_nf_send(&io,"\"tokens\":[",10))goto done;
    for(int off=0;off<prompt->len;) {
        size_t at=0;
        while(off<prompt->len&&at+16<sizeof(header)) {
            n=snprintf(header+at,sizeof(header)-at,"%s%d",off?",":"",prompt->v[off]);
            off++;
            if(n<0||(size_t)n>=sizeof(header)-at)goto done;at+=(size_t)n;
        }
        if(!ds4_nf_send(&io,header,at))goto done;
    }
    if(!ds4_nf_send(&io,"]",1))goto done;
    if(image_count) {
        if(image_count>16||!ds4_nf_send(&io,",\"vision\":[",11))goto done;
        static const char alphabet[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for(size_t i=0;i<image_count;i++) {
            const ds4_vision_span *image=&images[i];const ds4_vision_embedding *embedding=&image->embedding;
            if(!embedding->data||!embedding->token_count)goto done;
            char fingerprint[65];ds4_nf_digest(embedding->fingerprint,fingerprint);
            n=snprintf(header,sizeof(header),"%s{\"position\":%u,\"count\":%u,\"grid_height\":%u,\"grid_width\":%u,\"fingerprint\":\"%s\",\"data\":\"",
                i?",":"",image->token_start,embedding->token_count,embedding->grid_height,embedding->grid_width,fingerprint);
            if(n<0||(size_t)n>=sizeof(header)||!ds4_nf_send(&io,header,(size_t)n))goto done;
            const unsigned char *raw=(const unsigned char*)embedding->data;
            const uint64_t bytes=(uint64_t)embedding->token_count*ds4_engine_embd_dim(engine)*4u;
            char page[65536];size_t used=0;
            for(uint64_t off=0;off<bytes;off+=3) {
                const unsigned a=raw[off],b=off+1<bytes?raw[off+1]:0,d=off+2<bytes?raw[off+2]:0;
                page[used++]=alphabet[a>>2];page[used++]=alphabet[((a&3)<<4)|(b>>4)];
                page[used++]=off+1<bytes?alphabet[((b&15)<<2)|(d>>6)]:'=';
                page[used++]=off+2<bytes?alphabet[d&63]:'=';
                if(used==sizeof(page)){if(!ds4_nf_send(&io,page,used))goto done;used=0;}
            }
            if((used&&!ds4_nf_send(&io,page,used))||!ds4_nf_send(&io,"\"}",2))goto done;
        }
        if(!ds4_nf_send(&io,"]",1))goto done;
    }
    if(!ds4_nf_send(&io,"}\n",2))goto done;sent_at=ds4_nf_now();
    uint64_t work_layer=0,work_layers=0,work_rows=0;
    for(;;) {
        if(!ds4_qfn_line(&io,&reply)||!reply.ok)goto done;
        if(strcmp(reply.event,"prefill_progress"))break;
        if(reply.layer!=work_layer+1||!reply.layers||reply.layer>reply.layers||
           reply.layers>4096||(work_layers&&reply.layers!=work_layers)||!reply.work_rows||reply.work_rows>(uint64_t)prompt->len||
           (work_rows&&reply.work_rows!=work_rows)||(delta&&reply.work_rows!=(uint64_t)prompt->len-persistent->synced.position))goto done;
        work_layer=reply.layer;work_layers=reply.layers;work_rows=reply.work_rows;
        // Layer-major work does not imply completed token rows until all layers
        // and the output head finish. Report real work in its own progress unit.
        ds4_session_report_progress(session,"qfn_prefill_layer",(int)work_layer,(int)work_layers);
    }
    if(strcmp(reply.event,"prefilled")||(work_layer&&work_layer!=work_layers))goto done;
    fprintf(stderr,"QFN_PREFILL_PROGRESS layers=%llu rows=%d delta=%d\n",(unsigned long long)work_layer,prompt->len,delta);
    prefilled_at=ds4_nf_now();producer_seconds=reply.seconds;
    const double producer_prefill=reply.seconds,handover_start=ds4_nf_now();
    phase="callback_accept";fprintf(stderr,"QFN_STATE_PHASE side=mac phase=%s rows=%d\n",phase,prompt->len);
    if(listener>=0) {
        ds4_nf_io waiting=io;waiting.fd=listener;
        if(!ds4_nf_wait(&waiting,POLLIN))goto done;
        int data=accept(listener,NULL,NULL);if(data<0)goto done;
        int mode=fcntl(data,F_GETFL,0);if(mode<0||fcntl(data,F_SETFL,mode|O_NONBLOCK)){close(data);goto done;}
        control=io.fd;io.fd=data;close(listener);listener=-1;
        if(!ds4_qfn_line(&io,&reply)||!reply.ok||strcmp(reply.event,"state_paired")||
           strcmp(reply.nonce,nonce)||strcmp(reply.model,c->model_sha)||strcmp(reply.compatibility,c->compatibility_sha))goto done;
        size_t different=strlen(reply.token)^strlen(peer.token);
        for(size_t i=0;i<strlen(peer.token);i++)different|=(unsigned char)peer.token[i]^(i<strlen(reply.token)?(unsigned char)reply.token[i]:0);
        memset(reply.token,0,sizeof(reply.token));if(different)goto done;
    }
    phase="stream_descriptor";fprintf(stderr,"QFN_STATE_PHASE side=mac phase=%s rows=%d\n",phase,prompt->len);
    if(!ds4_qfn_line(&io,&reply)||!reply.ok||strcmp(reply.event,"state_stream")||
       strcmp(reply.model,c->model_sha)||strcmp(reply.compatibility,c->compatibility_sha)||
       strcmp(reply.encoding,delta?"qfn-delta-f32-v1":c->test_wide_index?"f32-index-v1":"f16-index-v1")||reply.rows!=(uint32_t)prompt->len||
       reply.rows>=(uint32_t)context||reply.mtp_rows>reply.rows||reply.verify!=c->verify_state||
       !reply.bytes||reply.bytes>c->payload_budget)goto done;
    const double export_prepare=reply.export_prepare_seconds;
    ds4_qfn_wire_stream stream={0};uint64_t canonical=0;
    int count=0;
    if(delta)canonical=reply.bytes;
    else count=ds4_qfn_native_layout(engine,(uint32_t)reply.rows,(uint32_t)reply.mtp_rows,&canonical,stream.begin,stream.end,256);
    if(count<0||canonical!=reply.canonical)goto done;uint64_t wire=canonical;
    if(!delta&&!c->test_wide_index)for(int i=0;i<count;i++)wire-=(stream.end[i]-stream.begin[i])/2;
    if(wire!=reply.bytes)goto done;
    stream.count=delta||c->test_wide_index?0u:(uint32_t)count;stream.wire=(ds4_ns_stream){&io,wire};stream.verify=c->verify_state;
    if(stream.verify)CC_SHA256_Init(&stream.digest);
    const double started=ds4_nf_now();
    mutated=true;phase="payload_load";fprintf(stderr,"QFN_STATE_PHASE side=mac phase=%s rows=%d wire_bytes=%llu\n",phase,prompt->len,(unsigned long long)wire);
    if(delta)rc=ds4_qfn_session_delta_apply(session,persistent->synced.position,(uint32_t)reply.rows,
        persistent->synced.mtp_rows,(uint32_t)reply.mtp_rows,wire,c->payload_budget,ds4_qfn_delta_read,&stream,err,cap);
    else {
        FILE *file=funopen(&stream,ds4_qfn_file_read,NULL,NULL,NULL);if(!file)goto done;
        rc=ds4_session_load_payload(session,file,canonical,err,cap);fclose(file);
    }
    const double loaded=ds4_nf_now();phase="stream_trailer";
    if(rc||stream.wire.remaining||stream.at!=canonical||!ds4_qfn_line(&io,&reply)||!reply.ok||
       strcmp(reply.event,"state_streamed")||reply.bytes!=wire){rc=-1;goto done;}
    if(stream.verify){unsigned char bytes[32];char hash[65];CC_SHA256_Final(bytes,&stream.digest);ds4_nf_digest(bytes,hash);
        if(strcmp(hash,reply.sha)){rc=-1;goto done;}}
    else if(reply.sha[0]){rc=-1;goto done;}
    const ds4_tokens *actual=ds4_session_tokens(session);
    if(!actual||actual->len!=prompt->len||ds4_session_pos(session)!=prompt->len||
       memcmp(actual->v,prompt->v,(size_t)prompt->len*4)){rc=-1;goto done;}
    if(!ds4_session_restore_vision_identities(session,images,image_count)){rc=-1;goto done;}
    if(!reply.have_checkpoint||reply.next_sequence!=persistent->sequence+1||!ds4_qfn_checkpoint_local(session,&reply.checkpoint)) {rc=-1;goto done;}
    const ds4_qfn_checkpoint imported=reply.checkpoint;
    const double gpu_read=reply.gpu_read_seconds,socket_send=reply.socket_send_seconds,hash_time=reply.hash_seconds,export_send=reply.export_send_seconds;
    phase="pair_owner";if(!delta&&ds4_qfn_session_pair(session,err,cap)){rc=-1;goto done;}
    if(control>=0){ds4_ns_disconnect(&io);io.fd=control;control=-1;}
    phase="checkpoint_ack";ds4_ns_text ack={header,sizeof(header),true};
    ds4_ns_put(&ack,"{\"op\":\"acknowledge\",\"session\":\"%s\",\"sequence\":%llu,\"checkpoint\":",persistent->id,(unsigned long long)reply.next_sequence);
    ds4_qfn_checkpoint_text(&ack,&imported);ds4_ns_put(&ack,"}\n");
    if(!ack.ok||!ds4_nf_send(&io,header,(size_t)(ack.p-header))||!ds4_qfn_line(&io,&reply)||!reply.ok||
       strcmp(reply.event,"acknowledged")||reply.next_sequence!=persistent->sequence+2||!reply.have_checkpoint||
       !ds4_qfn_checkpoint_same(&imported,&reply.checkpoint)){rc=-1;goto done;}
    persistent->synced=imported;persistent->sequence=reply.next_sequence;persistent->started=true;persistent->poisoned=false;
    persistent->captured_position=imported.position;persistent->captured_mtp=imported.mtp_rows;
    persistent->request_cached=delta?live:0;ds4_qfn_active_session=persistent;
    if(delta)persistent->forward_bytes+=wire;else persistent->initial_bytes+=wire;
    if(delta)fprintf(stderr,"QFN_FORWARD_DELTA base=%d target=%d bytes=%llu acknowledged=1\n",live,prompt->len,(unsigned long long)wire);
    fprintf(stderr,"QFN_NATIVE_IMPORT prefill_s=%.6f handover_s=%.6f export_prepare_s=%.6f rows=%d wire_bytes=%llu canonical_bytes=%llu stream_load_s=%.6f socket_read_s=%.6f inline_hash_s=%.6f restore_other_s=%.6f verify_state=%u producer_gpu_read_s=%.6f producer_socket_send_s=%.6f producer_inline_hash_s=%.6f producer_export_and_send_s=%.6f index_wire_bits=%u images=%zu transport=%s\n",
        producer_prefill,ds4_nf_now()-handover_start,export_prepare,prompt->len,(unsigned long long)wire,(unsigned long long)canonical,loaded-started,
        stream.read_seconds,stream.hash_seconds,loaded-started-stream.read_seconds-stream.hash_seconds,(unsigned)c->verify_state,
        gpu_read,socket_send,hash_time,export_send,delta||c->test_wide_index?32u:16u,image_count,c->state_listen_host?"direct":"ssh");
    rc=0;
done:
    if(listener>=0)close(listener);if(control>=0)close(control);
    if(!rc)fprintf(stderr,"QFN_REQUEST_PHASE rows=%d pairing_s=%.6f framing_send_s=%.6f prefill_wait_s=%.6f prefill_s=%.6f handover_s=%.6f total_s=%.6f transport=%s\n",
        prompt->len,paired_at-request_start,sent_at-paired_at,prefilled_at-sent_at,producer_seconds,
        ds4_nf_now()-prefilled_at,ds4_nf_now()-request_start,c->state_listen_host?"direct":"ssh");
    memset(peer.token,0,sizeof(peer.token));ds4_ns_disconnect(&io);
    if(rc){persistent->poisoned=true;if(mutated)ds4_session_invalidate(session);
        fprintf(stderr,"QFN_STATE_FAILURE phase=%s rows=%d last_event=%s error=%s\n",phase,prompt->len,reply.event,err[0]?err:reply.error);
        if(!err[0])snprintf(err,cap,"QFN native peer failed at %s: %s",phase,reply.error[0]?reply.error:"pairing, prefill, state layout, stream or cancellation");}
    return rc;
}
