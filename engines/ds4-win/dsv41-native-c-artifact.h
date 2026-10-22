#ifndef DSV41_NATIVE_C_ARTIFACT_H
#define DSV41_NATIVE_C_ARTIFACT_H
#include "dsv41-native-c-layout.h"

/* C adapter for llama-kv-artifact v2.1 and llama-dsv41-artifact.cpp.
 * Identity authentication and base/target SHA verification belong to the
 * session owner. Every callback is synchronous, forward-only and <=64 KiB. */
typedef int (*dsv41_c_io)(void*,void*,size_t);
typedef int (*dsv41_c_read_span)(void*,uint32_t,uint64_t,void*,size_t);
static void dsv41_c_put32(uint8_t*p,uint32_t v){for(unsigned i=0;i<4;++i)p[i]=(uint8_t)(v>>(i*8));}
static void dsv41_c_put64(uint8_t*p,uint64_t v){for(unsigned i=0;i<8;++i)p[i]=(uint8_t)(v>>(i*8));}
static uint32_t dsv41_c_get32(const uint8_t*p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static size_t dsv41_c_native_header_mode(uint32_t mode,uint32_t context,uint32_t position,uint8_t out[60]){
    const uint32_t words[15]={0x34565344,2,context,mode,128,128,context+1,position,40,512,128,129280,0x413431,position,position<128?position:128};
    if(mode!=1&&mode!=3)return 0;
    const unsigned count=mode==3?15:13;
    for(unsigned i=0;i<count;++i)dsv41_c_put32(out+i*4,words[i]);
    return count*4;
}
static void dsv41_c_native_header(uint32_t context,uint32_t position,uint8_t out[60]){
    (void)dsv41_c_native_header_mode(3,context,position,out);
}
static int dsv41_c_artifact_header(const dsv41_c_plan*p,uint64_t session,uint64_t model,uint8_t out[204]){
    if(!p||!out||!session||!model||(p->mode!=1&&p->mode!=3)||p->artifact_bytes<256)return 0;
    memset(out,0,204);memcpy(out,"KVARTIF1",8);out[8]=1;out[10]=10;
    dsv41_c_put32(out+12,40);dsv41_c_put32(out+16,p->context);dsv41_c_put32(out+20,p->target);
    dsv41_c_put32(out+28,0x380);dsv41_c_put64(out+32,p->artifact_bytes-204);
    dsv41_c_put32(out+44,UINT32_MAX);dsv41_c_put64(out+52,model);
    /* Preserve the existing serializer's unused sampling defaults exactly. */
    const uint32_t sampling[21]={0,0,0,0,40,64,0,0x3f733333,0x3d4ccccd,0x3f800000,0x3f4ccccd,0,0x3f800000,0x3f800000,0,0,0,0x3dcccccd,0xbf800000,0x40a00000,0x3dcccccd};
    for(unsigned i=0;i<21;++i)dsv41_c_put32(out+60+i*4,sampling[i]);
    dsv41_c_put32(out+144,UINT32_MAX);dsv41_c_put64(out+160,session);dsv41_c_put32(out+196,p->base);
    return 1;
}
static int dsv41_c_transfer(dsv41_c_io io,void*ud,void*data,size_t bytes){
    uint8_t*p=(uint8_t*)data;while(bytes){size_t n=bytes<65536?bytes:65536;if(!io(ud,p,n))return 0;p+=n;bytes-=n;}return 1;
}
static int dsv41_c_values_valid(const int32_t*t,uint32_t count,const float*logits){
    if(!t||!logits)return 0;
    for(uint32_t i=0;i<count;++i)if(t[i]<0||(uint32_t)t[i]>=129280)return 0;
    for(uint32_t i=0;i<129280;++i){uint32_t bits;memcpy(&bits,logits+i,4);if((bits&0x7f800000)==0x7f800000)return 0;}
    return 1;
}
static int dsv41_c_write_delta(const dsv41_c_plan*p,uint64_t session,uint64_t model,
        const int32_t*new_tokens,const float*logits,dsv41_c_read_span read_span,void*state,
        dsv41_c_io write,void*stream){
    uint8_t header[264],buffer[65536];
    if(!p||!read_span||!write||!dsv41_c_values_valid(new_tokens,p->target-p->base,logits)||
       !dsv41_c_artifact_header(p,session,model,header))return 0;
    const size_t nh=dsv41_c_native_header_mode(p->mode,p->context,p->target,header+204);
    if(!write(stream,header,204+nh)||
       !dsv41_c_transfer(write,stream,(void*)new_tokens,(size_t)(p->target-p->base)*4)||
       !dsv41_c_transfer(write,stream,(void*)logits,129280*4))return 0;
    for(uint32_t i=0;i<p->slice_count;++i){const dsv41_c_slice*s=p->slices+i;
        for(uint64_t off=0;off<s->bytes;){size_t n=(size_t)(s->bytes-off<sizeof(buffer)?s->bytes-off:sizeof(buffer));
            if(!read_span(state,s->span,s->offset+off,buffer,n)||!write(stream,buffer,n))return 0;off+=n;}}
    return 1;
}
/* Read the exact existing envelope before any session mutation. Buffers are
 * supplied by the budgeted caller; state slices remain on the stream. */
static int dsv41_c_read_delta_prefix_mode(uint32_t mode,uint32_t context,uint32_t base,uint32_t limit,uint64_t bytes,
        uint64_t session,uint64_t model,dsv41_c_io read,void*stream,
        int32_t*new_tokens,float*logits,dsv41_c_plan*out){
    uint8_t header[264],expected[264];dsv41_c_plan p;
    if(!out)return 0;memset(out,0,sizeof(*out));
    const size_t nh=mode==3?60:52;
    if((mode!=1&&mode!=3)||!read||!new_tokens||!logits||!limit||limit>49152||!read(stream,header,204+nh))return 0;
    const uint32_t target=dsv41_c_get32(header+20);
    if(!dsv41_c_delta_mode(mode,context,base,target,&p)||target-base>limit||p.artifact_bytes!=bytes||
       !dsv41_c_artifact_header(&p,session,model,expected))return 0;
    dsv41_c_native_header_mode(mode,context,target,expected+204);
    if(memcmp(header,expected,204+nh)||
       !dsv41_c_transfer(read,stream,new_tokens,(size_t)(target-base)*4)||
       !dsv41_c_transfer(read,stream,logits,129280*4)||!dsv41_c_values_valid(new_tokens,target-base,logits))return 0;
    *out=p;return 1;
}
static int dsv41_c_read_delta_prefix(uint32_t context,uint32_t base,uint32_t limit,uint64_t bytes,
        uint64_t session,uint64_t model,dsv41_c_io read,void*stream,
        int32_t*new_tokens,float*logits,dsv41_c_plan*out){
    return dsv41_c_read_delta_prefix_mode(3,context,base,limit,bytes,session,model,read,stream,new_tokens,logits,out);
}
#endif
