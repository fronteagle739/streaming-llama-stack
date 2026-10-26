#ifndef DS4_NATIVE_SESSION_JSON_H
#define DS4_NATIVE_SESSION_JSON_H
/* Included after the existing server JSON/native HTTP primitives. */
#include "ds4_native_checkpoint.h"
#define DS4_NS_JSON_CAP 65536
typedef struct {char *p;size_t left;bool ok;} ds4_ns_text;
static void ds4_ns_put(ds4_ns_text*b,const char*fmt,...){
    if(!b->ok)return;va_list ap;va_start(ap,fmt);int n=vsnprintf(b->p,b->left,fmt,ap);va_end(ap);
    if(n<0||(size_t)n>=b->left){b->ok=false;return;}b->p+=n;b->left-=(size_t)n;
}
static void ds4_ns_checkpoint_text(ds4_ns_text*b,const ds4_native_checkpoint*c){
    ds4_ns_put(b,"{\"context\":%u,\"position\":%u,\"mode\":3,\"capture_end\":%u,\"capture_length\":%u,\"tokens_sha256\":\"%s\",\"logits_sha256\":\"%s\",\"owner_sha256\":[",
        c->context,c->position,c->capture_end,c->capture_length,c->tokens_sha256,c->logits_sha256);
    for(uint32_t i=0;i<c->owner_count;i++)ds4_ns_put(b,"%s\"%s\"",i?",":"",c->owner_sha256[i]);
    ds4_ns_put(b,"]}");
}
static bool ds4_ns_checkpoint_same(const ds4_native_checkpoint*a,const ds4_native_checkpoint*b){
    if(a->context!=b->context||a->position!=b->position||a->capture_end!=b->capture_end||a->capture_length!=b->capture_length||
       a->owner_count!=b->owner_count||strcmp(a->tokens_sha256,b->tokens_sha256)||strcmp(a->logits_sha256,b->logits_sha256))return false;
    for(uint32_t i=0;i<a->owner_count;i++)if(strcmp(a->owner_sha256[i],b->owner_sha256[i]))return false;return true;
}
static bool ds4_ns_sha(const char**p,char out[65]){return ds4_nf_ascii(p,out,65)&&ds4_nf_hex(out,64);}
static bool ds4_ns_checkpoint_parse(const char**p,ds4_native_checkpoint*out){
    ds4_native_checkpoint c={0};unsigned seen=0;uint64_t mode=0;
    json_ws(p);if(*(*p)++!='{')return false;
    for(;;){char key[32];uint64_t value=0;unsigned bit=0;
        if(!ds4_nf_ascii(p,key,sizeof(key)))return false;json_ws(p);if(*(*p)++!=':')return false;
        if(!strcmp(key,"tokens_sha256")){bit=32;if(!ds4_ns_sha(p,c.tokens_sha256))return false;}
        else if(!strcmp(key,"logits_sha256")){bit=64;if(!ds4_ns_sha(p,c.logits_sha256))return false;}
        else if(!strcmp(key,"owner_sha256")){
            bit=128;json_ws(p);if(*(*p)++!='[')return false;
            for(;;){if(c.owner_count>=55||!ds4_ns_sha(p,c.owner_sha256[c.owner_count++]))return false;
                json_ws(p);if(**p==']'){(*p)++;break;}if(*(*p)++!=',')return false;}
        }else{
            if(!ds4_nf_u64(p,&value)||value>1048576)return false;
            if(!strcmp(key,"context")){bit=1;c.context=(uint32_t)value;}
            else if(!strcmp(key,"position")){bit=2;c.position=(uint32_t)value;}
            else if(!strcmp(key,"mode")){bit=4;mode=value;}
            else if(!strcmp(key,"capture_end")){bit=8;c.capture_end=(uint32_t)value;}
            else if(!strcmp(key,"capture_length")){bit=16;c.capture_length=(uint32_t)value;}
            else return false;
        }
        if(seen&bit)return false;seen|=bit;json_ws(p);if(**p=='}'){(*p)++;break;}if(*(*p)++!=',')return false;
    }
    if(seen!=255||mode!=3||c.context<2||!c.position||c.position>=c.context||c.capture_end!=c.position||
       c.capture_length!=(c.position<128?c.position:128)||c.owner_count!=(c.position%2?55:49))return false;
    *out=c;return true;
}
typedef struct {
    char session[33],model_sha256[65],state[24],status[24],kind[32],error[1024];
    uint64_t sequence,next_sequence,bytes,received_bytes,maximum_prefill_tokens,maximum_decode_tokens,control_host_bytes,maximum_decode_delta_tokens;
    uint64_t yoco_keep,yoco_state_version;
    ds4_native_checkpoint checkpoint,base,target;
    bool have_checkpoint,have_base,have_target,peer_synchronized,closed,rtx_decode,cleanup_complete;
    uint64_t seen;
} ds4_ns_reply;
/* Bound nesting before invoking the existing general JSON value skipper. */
static bool ds4_ns_json_depth(const char*s,size_t n){
    unsigned depth=0;bool quoted=false,escape=false;
    for(size_t i=0;i<n;i++){unsigned char c=(unsigned char)s[i];if(!c)return false;
        if(quoted){if(escape)escape=false;else if(c=='\\')escape=true;else if(c=='"')quoted=false;else if(c<32)return false;}
        else if(c=='"')quoted=true;else if(c=='{'||c=='['){if(++depth>16)return false;}
        else if(c=='}'||c==']'){if(!depth)return false;depth--;}}
    return !quoted&&!depth;
}
static bool ds4_ns_reply_parse(const char*text,size_t n,ds4_ns_reply*out){
    if(!text||!n||n>DS4_NS_JSON_CAP||!ds4_ns_json_depth(text,n))return false;
    ds4_ns_reply r={0};const char*p=text;char names[32][64];unsigned count=0;
    json_ws(&p);if(*p++!='{')return false;
    for(;;){char key[64];unsigned bit=31;bool ok=true;
        if(count>=32||!ds4_nf_ascii(&p,key,sizeof(key)))return false;
        for(unsigned i=0;i<count;i++)if(!strcmp(names[i],key))return false;
        strcpy(names[count++],key);json_ws(&p);if(*p++!=':')return false;
#define NS_STR(name,index) if(!strcmp(key,#name)){bit=index;ok=ds4_nf_ascii(&p,r.name,sizeof(r.name));}
#define NS_UINT(name,index) if(!strcmp(key,#name)){bit=index;ok=ds4_nf_u64(&p,&r.name);}
#define NS_BOOL(name,index) if(!strcmp(key,#name)){bit=index;ok=json_bool(&p,&r.name);}
        NS_STR(session,0) else NS_STR(model_sha256,1) else NS_STR(state,2) else NS_STR(status,3)
        else NS_STR(kind,4) else NS_STR(error,5) else NS_UINT(sequence,6) else NS_UINT(next_sequence,7)
        else NS_UINT(bytes,8) else NS_UINT(received_bytes,9) else NS_UINT(maximum_prefill_tokens,10)
        else NS_UINT(maximum_decode_tokens,11) else NS_UINT(control_host_bytes,12)
        else NS_BOOL(peer_synchronized,13) else NS_BOOL(closed,14)
        else NS_BOOL(cleanup_complete,19)
        else NS_UINT(maximum_decode_delta_tokens,20)
        else NS_UINT(yoco_keep,21) else NS_UINT(yoco_state_version,22)
        else if(!strcmp(key,"RTX_decode")){bit=15;ok=json_bool(&p,&r.rtx_decode);}
        else if(!strcmp(key,"checkpoint")){bit=16;ok=ds4_ns_checkpoint_parse(&p,&r.checkpoint);r.have_checkpoint=ok;}
        else if(!strcmp(key,"base")){bit=17;json_ws(&p);if(!strncmp(p,"null",4))p+=4;else{ok=ds4_ns_checkpoint_parse(&p,&r.base);r.have_base=ok;}}
        else if(!strcmp(key,"target")){bit=18;ok=ds4_ns_checkpoint_parse(&p,&r.target);r.have_target=ok;}
        else ok=json_skip_value(&p);
#undef NS_STR
#undef NS_UINT
#undef NS_BOOL
        if(!ok)return false;r.seen|=UINT64_C(1)<<bit;
        json_ws(&p);if(*p=='}'){p++;break;}if(*p++!=',')return false;
    }
    json_ws(&p);if((size_t)(p-text)!=n)return false;*out=r;return true;
}
#endif
