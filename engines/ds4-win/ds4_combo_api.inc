/* Windows target-only handoff glue around the existing native state API and
 * server renderer. Bulk state uses parent-owned, physically locked named RAM;
 * HTTP carries bounded authenticated control only. No checkpoint file path. */
#if defined(_WIN32) && !defined(DS4_NO_GPU)
#include "ds4_native_checkpoint.h"
typedef struct {
    HANDLE handle;uint8_t*data;uint64_t capacity,cursor;
}ds4_combo_mapping;
typedef struct {
    char session[33],mapping[128],sha[65];uint64_t epoch,sequence,capacity;
    char*body;ds4_tokens tokens;unsigned seen;
}ds4_combo_command;
static char ds4_combo_owner[33];
static uint64_t ds4_combo_epoch,ds4_combo_sequence;
static bool ds4_combo_claimed;
typedef struct {char session[33];uint64_t epoch;ds4_native_checkpoint checkpoint;}ds4_combo_journal;

static bool ds4_combo_hex(const char*s,size_t n){
    if(!s||strlen(s)!=n)return false;
    for(size_t i=0;i<n;i++)if(!((s[i]>='0'&&s[i]<='9')||(s[i]>='a'&&s[i]<='f')))return false;
    return true;
}
static bool ds4_combo_uint(const char**p,uint64_t*out){
    json_ws(p);if(**p<'0'||**p>'9')return false;uint64_t n=0;
    if(**p=='0'&&(*p)[1]>='0'&&(*p)[1]<='9')return false;
    while(**p>='0'&&**p<='9'){unsigned d=(unsigned)(*(*p)++-'0');if(n>(UINT64_MAX-d)/10)return false;n=n*10+d;}
    if(**p=='.'||**p=='e'||**p=='E')return false;*out=n;return true;
}
static bool ds4_combo_parse(const char*text,ds4_combo_command*c){
    memset(c,0,sizeof(*c));const char*p=text;json_ws(&p);if(*p++!='{')return false;
    json_ws(&p);if(*p=='}')return false;
    for(;;){char*key=NULL;unsigned bit=0;bool ok=false;
        if(!json_string(&p,&key))return false;json_ws(&p);if(*p++!=':'){free(key);return false;}
        if(!strcmp(key,"session")||!strcmp(key,"mapping")||!strcmp(key,"checkpoint_sha256")){
            char*value=NULL;ok=json_string(&p,&value);
            char*dst=!strcmp(key,"session")?c->session:!strcmp(key,"mapping")?c->mapping:c->sha;
            size_t cap=!strcmp(key,"session")?sizeof(c->session):!strcmp(key,"mapping")?sizeof(c->mapping):sizeof(c->sha);
            bit=!strcmp(key,"session")?1:!strcmp(key,"mapping")?16:32;
            if(ok&&strlen(value)<cap)strcpy(dst,value);else ok=false;free(value);
        }else if(!strcmp(key,"owner_epoch")||!strcmp(key,"sequence")||!strcmp(key,"capacity")){
            uint64_t*dst=!strcmp(key,"owner_epoch")?&c->epoch:!strcmp(key,"sequence")?&c->sequence:&c->capacity;
            bit=!strcmp(key,"owner_epoch")?2:!strcmp(key,"sequence")?4:64;ok=ds4_combo_uint(&p,dst);
        }else if(!strcmp(key,"request")){
            bit=8;json_ws(&p);const char*first=p;ok=*p=='{'&&json_skip_value(&p);
            if(ok){free(c->body);c->body=xstrndup(first,(size_t)(p-first));}
        }else if(!strcmp(key,"tokens")){
            bit=128;json_ws(&p);ok=*p++=='[';json_ws(&p);
            if(ok&&*p!=']')for(;;){uint64_t value;
                if(c->tokens.len>=49152||!ds4_combo_uint(&p,&value)||value>=129280){ok=false;break;}
                ds4_tokens_push(&c->tokens,(int)value);json_ws(&p);if(*p!=',')break;p++;}
            if(ok)ok=*p++==']';
        }
        free(key);if(!ok||!bit||(c->seen&bit))return false;c->seen|=bit;
        json_ws(&p);if(*p=='}'){p++;break;}if(*p++!=',')return false;
    }
    json_ws(&p);return !*p&&(c->seen&7)==7&&ds4_combo_hex(c->session,32)&&c->epoch&&c->sequence;
}
static void ds4_combo_mapping_close(ds4_combo_mapping*m){
    if(m->data)UnmapViewOfFile(m->data);if(m->handle)CloseHandle(m->handle);memset(m,0,sizeof(*m));
}
static bool ds4_combo_mapping_open(ds4_combo_mapping*m,const char*name,uint64_t capacity,bool write){
    memset(m,0,sizeof(*m));const char*prefix="Local\\DS4COMBO-";size_t n=strlen(prefix);
    if(!name||strncmp(name,prefix,n)||!ds4_combo_hex(name+n,64)||!capacity||capacity>2ull*1024*1024*1024)return false;
    wchar_t wide[128];if(!MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,name,-1,wide,128))return false;
    DWORD access=write?FILE_MAP_READ|FILE_MAP_WRITE:FILE_MAP_READ;
    m->handle=OpenFileMappingW(access,FALSE,wide);if(!m->handle)return false;
    m->data=MapViewOfFile(m->handle,access,0,0,(SIZE_T)capacity);m->capacity=capacity;
    if(!m->data){ds4_combo_mapping_close(m);return false;}return true;
}
static int ds4_combo_read(void*ud,void*dst,size_t n){ds4_combo_mapping*m=ud;
    if(n>65536||m->cursor>m->capacity||n>m->capacity-m->cursor)return 0;
    memcpy(dst,m->data+m->cursor,n);m->cursor+=n;return 1;
}
static bool ds4_combo_write(ds4_combo_mapping*m,const void*src,size_t n){
    if(n>65536||m->cursor>m->capacity||n>m->capacity-m->cursor)return false;
    memcpy(m->data+m->cursor,src,n);m->cursor+=n;return true;
}
static int ds4_combo_write_page(void*ud,void*src,size_t n){return ds4_combo_write(ud,src,n);}
static uint64_t ds4_combo_payload_bytes(const ds4_native_checkpoint*c){
    uint64_t bytes=52+(uint64_t)c->position*4+129280*4;
    for(uint32_t i=0;i<c->owner_count;++i)bytes+=c->owners[i].bytes;return bytes;
}
static void ds4_combo_describe(buf*b,const ds4_native_checkpoint*c){
    buf_printf(b,"{\"context\":%u,\"position\":%u,\"mode\":1,\"capture_end\":0,\"capture_length\":0,\"tokens_sha256\":\"%s\",\"logits_sha256\":\"%s\",\"owner_sha256\":[",c->context,c->position,c->tokens_sha256,c->logits_sha256);
    for(uint32_t i=0;i<c->owner_count;++i)buf_printf(b,"%s\"%s\"",i?",":"",c->owner_sha256[i]);
    buf_printf(b,"],\"payload_sha256\":\"%s\",\"bytes\":%llu,\"owners\":[",c->payload_sha256,(unsigned long long)ds4_combo_payload_bytes(c));
    static const char*names[]={"dspark","raw","compressed","index","pool_kv","pool_score"};
    for(uint32_t i=0;i<c->owner_count;++i)buf_printf(b,"%s{\"kind\":\"%s\",\"index\":%u,\"bytes\":%llu,\"sha256\":\"%s\"}",i?",":"",names[c->owners[i].kind],c->owners[i].index,(unsigned long long)c->owners[i].bytes,c->owner_sha256[i]);
    buf_puts(b,"]}");
}
static int ds4_combo_export_view(void*ud,const ds4_native_view*v,char*err,size_t cap){
    (void)err;(void)cap;ds4_combo_mapping*m=ud;uint8_t header[60],page[65536];
    size_t n=dsv41_c_native_header_mode(1,v->context,v->position,header);
    if(!ds4_combo_write(m,header,n)||!dsv41_c_transfer(ds4_combo_write_page,m,(void*)v->tokens,(size_t)v->position*4)||!dsv41_c_transfer(ds4_combo_write_page,m,(void*)v->logits,129280*4))return 0;
    for(uint32_t i=0;i<v->span_count;++i)for(uint64_t off=0;off<v->spans[i].bytes;){
        size_t count=(size_t)(v->spans[i].bytes-off);if(count>65536)count=65536;
        if(!v->read_span(v->state,i,off,page,count)||!ds4_combo_write(m,page,count))return 0;off+=count;
    }return 1;
}
static int ds4_combo_anchor_view(void*ud,const ds4_native_view*v,char*err,size_t cap){
    ds4_combo_mapping*m=ud;uint8_t page[65536];ds4_combo_journal journal={0};
    strcpy(journal.session,ds4_combo_owner);journal.epoch=ds4_combo_epoch;
    if(!ds4_native_capture(&journal.checkpoint,v,err,cap)||!ds4_combo_write(m,&journal,sizeof(journal))||!dsv41_c_transfer(ds4_combo_write_page,m,(void*)v->logits,129280*4))return 0;
    for(uint32_t i=0;i<v->span_count;++i){
        if(v->spans[i].kind==DSV41_C_COMPRESSED||v->spans[i].kind==DSV41_C_INDEX)continue;
        for(uint64_t off=0;off<v->spans[i].bytes;){size_t n=(size_t)(v->spans[i].bytes-off);if(n>65536)n=65536;
            if(!v->read_span(v->state,i,off,page,n)||!ds4_combo_write(m,page,n))return 0;off+=n;}
    }return 1;
}
static int ds4_combo_verify_sha(void*ud,const ds4_native_view*v,char*err,size_t cap){
    ds4_native_checkpoint c={0};return ds4_native_capture(&c,v,err,cap)&&!strcmp(c.payload_sha256,(const char*)ud);
}
static int ds4_combo_restore_mapping(ds4_session*s,const char*name,uint64_t capacity,const char*sha,char*err,size_t cap){
    ds4_combo_mapping m={0};int rc=1;int32_t*tokens=NULL;float*logits=NULL;uint32_t header[13];
    if(!ds4_combo_hex(sha,64)||!ds4_combo_mapping_open(&m,name,capacity,false)||!ds4_combo_read(&m,header,52))goto done;
    uint32_t position=header[7],context=(uint32_t)ds4_session_ctx(s);dsv41_c_span spans[55];uint32_t count=0;
    uint8_t expected[60];dsv41_c_native_header_mode(1,context,position,expected);
    if(memcmp(expected,header,52)||!dsv41_c_layout_mode(1,context,position,spans,&count))goto done;
    uint64_t bytes=52+(uint64_t)position*4+129280*4;for(uint32_t i=0;i<count;i++)bytes+=spans[i].bytes;
    if(bytes>capacity)goto done;
    tokens=malloc((size_t)position*4);logits=malloc(129280*4);if(!tokens||!logits)goto done;
    for(size_t off=0;off<(size_t)position*4;){size_t n=(size_t)position*4-off;if(n>65536)n=65536;if(!ds4_combo_read(&m,(uint8_t*)tokens+off,n))goto done;off+=n;}
    for(size_t off=0;off<129280*4;){size_t n=129280*4-off;if(n>65536)n=65536;if(!ds4_combo_read(&m,(uint8_t*)logits+off,n))goto done;off+=n;}
    rc=ds4_session_native_restore(s,position,tokens,logits,ds4_combo_read,&m,ds4_combo_verify_sha,(void*)sha,err,cap);
    if(!rc&&m.cursor!=bytes)rc=1;
done:
    free(tokens);free(logits);ds4_combo_mapping_close(&m);
    if(rc&&err&&cap&&!err[0])snprintf(err,cap,"native shared-RAM checkpoint refused");return rc;
}
/* Qualification only: full vocabulary vectors in parent-owned locked RAM.
 * Compare on the SAME reference prefix, so KL is not confounded by drift. */
#define DS4_COMBO_PROBE_ROWS 384u
#define DS4_COMBO_PROBE_VOCAB 129280u
static int ds4_combo_probe(ds4_session*s,ds4_combo_mapping*m,bool reference,buf*reply,char*err,size_t cap){
    const uint64_t row_bytes=DS4_COMBO_PROBE_VOCAB*4ull;
    const uint64_t bytes=32+DS4_COMBO_PROBE_ROWS*4ull+DS4_COMBO_PROBE_ROWS*row_bytes;
    uint64_t header[4]={0x3145424f52503444ull,ds4_combo_epoch,DS4_COMBO_PROBE_ROWS,DS4_COMBO_PROBE_VOCAB};
    if(m->capacity!=bytes){snprintf(err,cap,"exact384-vector reference RAM extent required");return 1;}
    if(reference)memcpy(m->data,header,32);
    else if(memcmp(m->data,header,32)){snprintf(err,cap,"reference bank owner epoch/header differs");return 1;}
    float*row=malloc((size_t)row_bytes);if(!row){snprintf(err,cap,"probe row allocation refused");return 1;}
    int32_t*ids=(int32_t*)(m->data+32);double sum_kl=0,max_kl=0;unsigned matches=0,exact=0;int rc=1;
    for(unsigned r=0;r<DS4_COMBO_PROBE_ROWS;r++){
        if(ds4_session_copy_logits(s,row,DS4_COMBO_PROBE_VOCAB)!=DS4_COMBO_PROBE_VOCAB)goto done;
        int candidate=ds4_session_argmax(s);float*base=(float*)(m->data+32+DS4_COMBO_PROBE_ROWS*4ull+r*row_bytes);
        if(reference){ids[r]=candidate;memcpy(base,row,(size_t)row_bytes);}
        else{
            if(ids[r]<0||ids[r]>=DS4_COMBO_PROBE_VOCAB)goto done;
            matches+=(candidate==ids[r]);exact+=!memcmp(base,row,(size_t)row_bytes);
            double bm=-INFINITY,cm=-INFINITY,bz=0,cz=0,kl=0;
            for(unsigned v=0;v<DS4_COMBO_PROBE_VOCAB;v++){
                if(!isfinite(base[v])||!isfinite(row[v]))goto done;
                if(base[v]>bm)bm=base[v];if(row[v]>cm)cm=row[v];
            }
            for(unsigned v=0;v<DS4_COMBO_PROBE_VOCAB;v++){bz+=exp((double)base[v]-bm);cz+=exp((double)row[v]-cm);}
            const double bn=bm+log(bz),cn=cm+log(cz);
            for(unsigned v=0;v<DS4_COMBO_PROBE_VOCAB;v++)kl+=exp((double)base[v]-bn)*((double)base[v]-bn-(double)row[v]+cn);
            if(!isfinite(kl))goto done;sum_kl+=kl;if(kl>max_kl)max_kl=kl;
        }
        if(ds4_session_eval(s,ids[r],err,cap))goto done;
    }
    char digest[65];if(!ds4_native_hash(m->data,(size_t)bytes,digest))goto done;
    buf_printf(reply,"{\"vectors\":384,\"reference_sha256\":\"%s\",\"top1_matches\":%u,\"byte_exact_vectors\":%u,\"mean_kl\":%.17g,\"max_kl\":%.17g,",digest,reference?384:matches,reference?384:exact,reference?0:sum_kl/384,reference?0:max_kl);rc=0;
done:free(row);if(rc&&err&&cap&&!*err)snprintf(err,cap,"full-vector qualification failed");return rc;
}
static bool ds4_combo_start(server*s,const ds4_engine_options*opt,char*err,size_t cap){
    const char*enabled=getenv("DS4_COMBO_ENABLED");if(!enabled||!strcmp(enabled,"0"))return true;
    const char*id=getenv("DS4_COMBO_OWNER_SESSION"),*epoch=getenv("DS4_COMBO_OWNER_EPOCH"),*seq=getenv("DS4_COMBO_SEQUENCE");
    const char*token=getenv("DS4_SERVER_AUTH_TOKEN");
    if(strcmp(enabled,"1")||!ds4_combo_hex(id,32)||!epoch||!seq||!token||strlen(token)<32||opt->backend!=DS4_BACKEND_CUDA||
       opt->dspark||opt->tp.role!=DS4_TP_NONE||opt->distributed.role!=DS4_DISTRIBUTED_NONE||s->slot_count!=1||s->batched_mode||s->ctx_size>262144||s->training_features)goto bad;
    const char*p=epoch;if(!ds4_combo_uint(&p,&ds4_combo_epoch)||*p||!ds4_combo_epoch)goto bad;
    p=seq;if(!ds4_combo_uint(&p,&ds4_combo_sequence)||*p||!ds4_combo_sequence)goto bad;
    strcpy(ds4_combo_owner,id);ds4_combo_active=true;ds4_combo_claimed=false;
    const char*name=getenv("DS4_COMBO_RESTORE_MAPPING"),*size=getenv("DS4_COMBO_RESTORE_CAPACITY"),*sha=getenv("DS4_COMBO_RESTORE_SHA256");
    if(name||size||sha){uint64_t capacity=0;if(!name||!size||!sha)goto bad;p=size;if(!ds4_combo_uint(&p,&capacity)||*p||
        ds4_combo_restore_mapping(s->slots[0].session,name,capacity,sha,err,cap))goto bad;}
    fprintf(stderr,"DS4COMBO_NATIVE_READY context=%d restored_position=%d epoch=%llu RAM_state_only=1\n",s->ctx_size,ds4_session_pos(s->slots[0].session),(unsigned long long)ds4_combo_epoch);return true;
bad:if(err&&cap&&!err[0])snprintf(err,cap,"explicit target-only combo configuration refused");return false;
}
static void ds4_combo_http(server*s,int fd,const http_request*hr){
    char err[512]={0};ds4_combo_command command={0};buf reply={0};ds4_combo_mapping mapping={0};
    if(strcmp(hr->method,"POST")||!hr->quality_authorized){http_error(fd,s->enable_cors,401,"authenticated POST and owner epoch required");return;}
    if(!hr->body_len||hr->body_len>(1u<<20)||!ds4_combo_parse(hr->body,&command)){http_error(fd,s->enable_cors,400,"bounded exact combo control required");goto done;}
    pthread_mutex_lock(&s->inference_mu);
    if(strcmp(command.session,ds4_combo_owner)||command.epoch!=ds4_combo_epoch||command.sequence!=ds4_combo_sequence||ds4_combo_sequence==UINT64_MAX){snprintf(err,sizeof(err),"stale combo owner/epoch/sequence");goto failed;}
    ds4_session*session=s->slots[0].session;ds4_native_checkpoint current={0};const char*path=hr->path;
    if(!strcmp(path,"/decode/setup")){
        if(command.seen!=7||ds4_combo_claimed){snprintf(err,sizeof(err),"decoder setup ownership already claimed");goto failed;}
        ds4_combo_claimed=true;
    }else{
        if(!ds4_combo_claimed){snprintf(err,sizeof(err),"decoder setup ownership required");goto failed;}
        if(!strcmp(path,"/decode/render")){
            if(command.seen!=(7|8)){snprintf(err,sizeof(err),"render request fields differ");goto failed;}
            request parsed;char why[256]={0};
            if(!parse_chat_request(s->engine,s,command.body,1,s->ctx_size,&parsed,why,sizeof(why))){snprintf(err,sizeof(err),"native renderer refused: %.200s",why);goto failed;}
            buf_puts(&reply,"{\"tokens\":[");for(int i=0;i<parsed.prompt.len;i++)buf_printf(&reply,"%s%d",i?",":"",parsed.prompt.v[i]);buf_puts(&reply,"],");request_free(&parsed);
        }else if(!strcmp(path,"/decode/sync")){
            if(command.seen!=(7|32|128)||!command.tokens.len){snprintf(err,sizeof(err),"sync fields/extent differ");goto failed;}
            const ds4_tokens*old=ds4_session_tokens(session);uint32_t base=(uint32_t)ds4_session_pos(session);
            if(base&&(ds4_session_native_visit(session,ds4_native_capture,&current,err,sizeof(err))||strcmp(current.payload_sha256,command.sha)))goto failed;
            if(!base&&*command.sha){snprintf(err,sizeof(err),"empty decoder base identity differs");goto failed;}
            if((uint64_t)base+command.tokens.len>=(uint64_t)s->ctx_size){snprintf(err,sizeof(err),"native sync context exhausted");goto failed;}
            ds4_tokens full={0};if(base)ds4_tokens_copy(&full,old);for(int i=0;i<command.tokens.len;i++)ds4_tokens_push(&full,command.tokens.v[i]);
            int rc=ds4_session_sync(session,&full,err,sizeof(err));ds4_tokens_free(&full);if(rc)goto failed;
        }else{
            if(ds4_session_native_visit(session,ds4_native_capture,&current,err,sizeof(err)))goto failed;
            if(!strcmp(path,"/decode/checkpoint")){
                if(command.seen!=7){snprintf(err,sizeof(err),"checkpoint fields differ");goto failed;}
            }else if(!strcmp(path,"/decode/reference")||!strcmp(path,"/decode/compare")){
                bool reference=!strcmp(path,"/decode/reference");
                if(command.seen!=(7|16|32|64)||strcmp(command.sha,current.payload_sha256)||!ds4_combo_mapping_open(&mapping,command.mapping,command.capacity,reference)||ds4_combo_probe(session,&mapping,reference,&reply,err,sizeof(err)))goto failed;
            }else if(!strcmp(path,"/decode/export")||!strcmp(path,"/decode/anchor")){
                if(command.seen!=(7|16|32|64)||strcmp(command.sha,current.payload_sha256)||!ds4_combo_mapping_open(&mapping,command.mapping,command.capacity,true)){snprintf(err,sizeof(err),"native export mapping/base refused");goto failed;}
                bool anchor=!strcmp(path,"/decode/anchor");
                if(anchor){if(command.capacity>24u*1024*1024||ds4_session_native_visit(session,ds4_combo_anchor_view,&mapping,err,sizeof(err)))goto failed;
                }else if(command.capacity<ds4_combo_payload_bytes(&current)||ds4_session_native_visit(session,ds4_combo_export_view,&mapping,err,sizeof(err)))goto failed;
                buf_printf(&reply,"{\"written_bytes\":%llu,",(unsigned long long)mapping.cursor);
            }else if(!strcmp(path,"/decode/rewind")){
                ds4_combo_journal journal={0};
                if(command.seen!=(7|16|32|64)||command.capacity>24u*1024*1024||!ds4_combo_mapping_open(&mapping,command.mapping,command.capacity,false)||!ds4_combo_read(&mapping,&journal,sizeof(journal))||memcmp(journal.session,ds4_combo_owner,33)||journal.epoch!=ds4_combo_epoch||strcmp(journal.checkpoint.payload_sha256,command.sha)){snprintf(err,sizeof(err),"saved native rewind owner/epoch/checkpoint unavailable");goto failed;}
                float*logits=(float*)(mapping.data+sizeof(journal));mapping.cursor=sizeof(journal)+129280*4;
                if(mapping.cursor>mapping.capacity||ds4_session_native_rewind(session,journal.checkpoint.position,logits,ds4_combo_read,&mapping,ds4_native_verify,&current,ds4_native_verify,&journal.checkpoint,err,sizeof(err)))goto failed;
            }else{snprintf(err,sizeof(err),"unknown combo operation");goto failed;}
        }
    }
    if(!reply.len)buf_puts(&reply,"{");
    ds4_combo_sequence++;
    buf_printf(&reply,"\"session\":\"%s\",\"owner_epoch\":%llu,\"next_sequence\":%llu,\"status\":\"complete\",\"checkpoint\":",ds4_combo_owner,(unsigned long long)ds4_combo_epoch,(unsigned long long)ds4_combo_sequence);
    if(ds4_session_pos(session)>0){if(ds4_session_native_visit(session,ds4_native_capture,&current,err,sizeof(err)))goto failed;ds4_combo_describe(&reply,&current);}else buf_puts(&reply,"null");
    buf_puts(&reply,"}\n");pthread_mutex_unlock(&s->inference_mu);
    http_response(fd,s->enable_cors,200,"application/json",reply.ptr);goto done;
failed:
    pthread_mutex_unlock(&s->inference_mu);http_error(fd,s->enable_cors,409,*err?err:"combo state verification refused");
done:
    free(command.body);ds4_tokens_free(&command.tokens);buf_free(&reply);ds4_combo_mapping_close(&mapping);
}
#else
static bool ds4_combo_start(server*s,const ds4_engine_options*o,char*e,size_t n){(void)s;(void)o;(void)e;(void)n;const char*v=getenv("DS4_COMBO_ENABLED");return !v||!strcmp(v,"0");}
static void ds4_combo_http(server*s,int fd,const http_request*r){(void)r;http_error(fd,s->enable_cors,409,"combo requires native Windows CUDA");}
#endif
