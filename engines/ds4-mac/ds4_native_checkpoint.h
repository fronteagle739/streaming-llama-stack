#ifndef DS4_NATIVE_CHECKPOINT_H
#define DS4_NATIVE_CHECKPOINT_H
#include "ds4_native_session.h"
#include "dsv41-native-c-artifact.h"
#include <CommonCrypto/CommonDigest.h>

typedef struct {
    uint32_t context,position,capture_end,capture_length,owner_count;
    char tokens_sha256[65],logits_sha256[65],owner_sha256[55][65],payload_sha256[65];
} ds4_native_checkpoint;
static void ds4_native_digest(const unsigned char d[32],char out[65]){
    static const char hex[]="0123456789abcdef";
    for(unsigned i=0;i<32;++i){out[i*2]=hex[d[i]>>4];out[i*2+1]=hex[d[i]&15];}out[64]=0;
}
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
static void ds4_native_hash(const void*data,size_t n,char out[65]){
    unsigned char digest[32];CC_SHA256(data,(CC_LONG)n,digest);ds4_native_digest(digest,out);
}
static int ds4_native_capture(void*ud,const ds4_native_view*v,char*err,size_t cap){
    ds4_native_checkpoint*out=ud;memset(out,0,sizeof(*out));
    if(!v||!v->read_span||v->span_count>55||!dsv41_c_values_valid(v->tokens,v->position,v->logits)){
        snprintf(err,cap,"native checkpoint invalid values");return 0;}
    ds4_native_checkpoint c={0};c.context=v->context;c.position=v->position;
    c.capture_end=v->capture_end;c.capture_length=v->capture_length;c.owner_count=v->span_count;
    uint8_t header[60],page[65536];unsigned char digest[32];CC_SHA256_CTX full;
    dsv41_c_native_header(v->context,v->position,header);CC_SHA256_Init(&full);CC_SHA256_Update(&full,header,60);
    ds4_native_hash(v->tokens,(size_t)v->position*4,c.tokens_sha256);
    ds4_native_hash(v->logits,129280*4,c.logits_sha256);
    CC_SHA256_Update(&full,v->tokens,v->position*4);CC_SHA256_Update(&full,v->logits,129280*4);
    for(uint32_t i=0;i<v->span_count;++i){CC_SHA256_CTX owner;CC_SHA256_Init(&owner);
        for(uint64_t off=0;off<v->spans[i].bytes;){size_t n=(size_t)(v->spans[i].bytes-off<sizeof(page)?v->spans[i].bytes-off:sizeof(page));
            if(!v->read_span(v->state,i,off,page,n)){snprintf(err,cap,"native checkpoint owner read failed");return 0;}
            CC_SHA256_Update(&owner,page,(CC_LONG)n);CC_SHA256_Update(&full,page,(CC_LONG)n);off+=n;}
        CC_SHA256_Final(digest,&owner);ds4_native_digest(digest,c.owner_sha256[i]);}
    CC_SHA256_Final(digest,&full);ds4_native_digest(digest,c.payload_sha256);*out=c;return 1;
}
#pragma clang diagnostic pop
#endif
