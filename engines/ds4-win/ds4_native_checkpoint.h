#ifndef DS4_NATIVE_CHECKPOINT_H
#define DS4_NATIVE_CHECKPOINT_H
#include "ds4_native_session.h"
#include "dsv41-native-c-artifact.h"
#include <windows.h>
#include <bcrypt.h>
#include <limits.h>

/* Windows port of the existing Mac checkpoint hasher. All owner reads remain
 * <=64KiB; hash identity is explicit kind/index rather than an array offset. */
typedef struct {
    uint32_t context,position,mode,owner_count;
    dsv41_c_span owners[55];
    char tokens_sha256[65],logits_sha256[65],owner_sha256[55][65],payload_sha256[65];
} ds4_native_checkpoint;
typedef struct { BCRYPT_ALG_HANDLE algorithm; BCRYPT_HASH_HANDLE hash; } ds4_native_sha;
static int ds4_native_sha_begin(ds4_native_sha*s){
    memset(s,0,sizeof(*s));
    if(BCryptOpenAlgorithmProvider(&s->algorithm,BCRYPT_SHA256_ALGORITHM,NULL,0)<0)return 0;
    if(BCryptCreateHash(s->algorithm,&s->hash,NULL,0,NULL,0,0)<0){BCryptCloseAlgorithmProvider(s->algorithm,0);s->algorithm=NULL;return 0;}
    return 1;
}
static void ds4_native_sha_close(ds4_native_sha*s){
    if(s->hash)BCryptDestroyHash(s->hash);
    if(s->algorithm)BCryptCloseAlgorithmProvider(s->algorithm,0);
    memset(s,0,sizeof(*s));
}
static int ds4_native_sha_add(ds4_native_sha*s,const void*data,size_t count){
    if(count>ULONG_MAX)return 0;
    return BCryptHashData(s->hash,(PUCHAR)data,(ULONG)count,0)>=0;
}
static int ds4_native_sha_end(ds4_native_sha*s,char out[65]){
    unsigned char digest[32];static const char hex[]="0123456789abcdef";
    int ok=BCryptFinishHash(s->hash,digest,32,0)>=0;
    ds4_native_sha_close(s);if(!ok)return 0;
    for(unsigned i=0;i<32;++i){out[2*i]=hex[digest[i]>>4];out[2*i+1]=hex[digest[i]&15];}
    out[64]=0;return 1;
}
static int ds4_native_hash(const void*data,size_t n,char out[65]){
    ds4_native_sha s;if(!ds4_native_sha_begin(&s))return 0;
    if(!ds4_native_sha_add(&s,data,n)){ds4_native_sha_close(&s);return 0;}
    return ds4_native_sha_end(&s,out);
}
static int ds4_native_capture(void*ud,const ds4_native_view*v,char*err,size_t cap){
    ds4_native_checkpoint*out=ud,c={0};ds4_native_sha full={0},owner={0};uint8_t header[60],page[65536];
    if(!out||!v||!v->read_span||v->span_count>55||!dsv41_c_values_valid(v->tokens,v->position,v->logits))goto failed;
    c.context=v->context;c.position=v->position;c.mode=1;c.owner_count=v->span_count;
    memcpy(c.owners,v->spans,v->span_count*sizeof(*v->spans));
    size_t nh=dsv41_c_native_header_mode(1,v->context,v->position,header);
    if(!ds4_native_sha_begin(&full)||!ds4_native_sha_add(&full,header,nh)||
       !ds4_native_hash(v->tokens,(size_t)v->position*4,c.tokens_sha256)||
       !ds4_native_hash(v->logits,129280*4,c.logits_sha256)||
       !ds4_native_sha_add(&full,v->tokens,(size_t)v->position*4)||
       !ds4_native_sha_add(&full,v->logits,129280*4))goto failed;
    for(uint32_t i=0;i<v->span_count;++i){
        if(!ds4_native_sha_begin(&owner))goto failed;
        for(uint64_t off=0;off<v->spans[i].bytes;){
            size_t n=(size_t)(v->spans[i].bytes-off);if(n>sizeof(page))n=sizeof(page);
            if(!v->read_span(v->state,i,off,page,n)||!ds4_native_sha_add(&owner,page,n)||!ds4_native_sha_add(&full,page,n))goto failed;
            off+=n;
        }
        if(!ds4_native_sha_end(&owner,c.owner_sha256[i]))goto failed;
    }
    if(!ds4_native_sha_end(&full,c.payload_sha256))goto failed;
    *out=c;return 1;
failed:
    ds4_native_sha_close(&owner);ds4_native_sha_close(&full);
    if(err&&cap)snprintf(err,cap,"native checkpoint hashing failed");return 0;
}
static int ds4_native_verify(void*ud,const ds4_native_view*v,char*err,size_t cap){
    const ds4_native_checkpoint*expected=ud;ds4_native_checkpoint actual={0};
    if(!expected||!ds4_native_capture(&actual,v,err,cap))return 0;
    if(expected->mode!=1||expected->context!=actual.context||expected->position!=actual.position||
       expected->owner_count!=actual.owner_count||strcmp(expected->tokens_sha256,actual.tokens_sha256)||
       strcmp(expected->logits_sha256,actual.logits_sha256)||strcmp(expected->payload_sha256,actual.payload_sha256))goto mismatch;
    for(uint32_t i=0;i<actual.owner_count;++i){uint32_t j;
        if(!dsv41_c_find(expected->owners,expected->owner_count,actual.owners[i].kind,actual.owners[i].index,&j)||
           expected->owners[j].bytes!=actual.owners[i].bytes||strcmp(expected->owner_sha256[j],actual.owner_sha256[i]))goto mismatch;
    }
    return 1;
mismatch:
    if(err&&cap)snprintf(err,cap,"native checkpoint identity mismatch");return 0;
}
#endif
