#ifndef DSV41_NATIVE_C_LAYOUT_H
#define DSV41_NATIVE_C_LAYOUT_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* C adapter for the existing native mode3 layout/delta plan. ds4 is C-only;
 * compare this mechanically against llama-dsv41-wire/delta in the fixture.
 * These are span descriptions, not buffers or a different network format. */
enum dsv41_c_kind { DSV41_C_DSPARK, DSV41_C_RAW, DSV41_C_COMPRESSED,
    DSV41_C_INDEX, DSV41_C_POOL_KV, DSV41_C_POOL_SCORE };
typedef struct { uint32_t kind,index; uint64_t bytes; } dsv41_c_span;
typedef struct { uint32_t span; uint64_t offset,bytes; } dsv41_c_slice;
typedef struct {
    uint32_t context,base,target,span_count,slice_count,mode;
    uint64_t state_bytes,artifact_bytes;
    dsv41_c_span spans[55];
    dsv41_c_slice slices[96];
} dsv41_c_plan;
static dsv41_c_span dsv41_c_span_value(uint32_t kind,uint32_t index,uint64_t bytes){
    dsv41_c_span s={kind,index,bytes};return s;
}
static int dsv41_c_layout_mode(uint32_t mode,uint32_t context,uint32_t position,dsv41_c_span out[55],uint32_t*count){
    if(count)*count=0;
    if(!out||!count||(mode!=1&&mode!=3)||!context||context>1048576||!position||position>=context)return 0;
    uint32_t n=0;const uint64_t raw=position<128?position:128;
    if(mode==3)out[n++]=dsv41_c_span_value(DSV41_C_DSPARK,0,raw*3*5120*4);
    for(uint32_t i=0;i<40;++i)out[n++]=dsv41_c_span_value(DSV41_C_RAW,i,raw*512*4);
    for(uint32_t i=0;i<4;++i){const uint64_t live=position/(i<3?2:1);
        out[n++]=dsv41_c_span_value(DSV41_C_COMPRESSED,i,live*512*4);
        out[n++]=dsv41_c_span_value(DSV41_C_INDEX,i,live*128*4);
        if(i<3&&(position&1)){out[n++]=dsv41_c_span_value(DSV41_C_POOL_KV,i,512*4);out[n++]=dsv41_c_span_value(DSV41_C_POOL_SCORE,i,512*4);}}
    *count=n;return n<=55;
}
static int dsv41_c_layout(uint32_t context,uint32_t position,dsv41_c_span out[55],uint32_t*count){
    return dsv41_c_layout_mode(3,context,position,out,count);
}
/* Resolve identity explicitly; mode1 omits DSpark and pooling owners appear
 * only at odd positions. Never transfer an ordinal between layouts. */
static int dsv41_c_find(const dsv41_c_span*spans,uint32_t count,uint32_t kind,uint32_t index,uint32_t*out){
    if(!spans||!out||count>55)return 0;
    uint32_t matches=0;
    for(uint32_t i=0;i<count;++i)if(spans[i].kind==kind&&spans[i].index==index){*out=i;++matches;}
    return matches==1;
}
static int dsv41_c_add(dsv41_c_plan*p,uint32_t span,uint64_t offset,uint64_t bytes){
    if(!bytes)return 1;
    if(p->slice_count>=96||span>=p->span_count||offset>p->spans[span].bytes||bytes>p->spans[span].bytes-offset||(offset&3)||(bytes&3))return 0;
    dsv41_c_slice s={span,offset,bytes};p->slices[p->slice_count++]=s;p->state_bytes+=bytes;return 1;
}
static int dsv41_c_delta_mode(uint32_t mode,uint32_t context,uint32_t base,uint32_t target,dsv41_c_plan*out){
    if(!out)return 0;memset(out,0,sizeof(*out));
    if(!base||base>=target||target-base>49152)return 0;
    dsv41_c_plan p;memset(&p,0,sizeof(p));p.context=context;p.base=base;p.target=target;p.mode=mode;
    if(!dsv41_c_layout_mode(mode,context,target,p.spans,&p.span_count))return 0;
    const uint32_t added=target-base;
    for(uint32_t i=0;i<p.span_count;++i){const dsv41_c_span*s=&p.spans[i];
        if(s->kind==DSV41_C_DSPARK||s->kind==DSV41_C_RAW){
            const uint64_t width=s->kind==DSV41_C_DSPARK?3u*5120u*4u:512u*4u;
            if(added>=128){if(!dsv41_c_add(&p,i,0,s->bytes))return 0;}
            else{const uint32_t first=base%128,count=added<128-first?added:128-first;
                if(!dsv41_c_add(&p,i,first*width,count*width)||!dsv41_c_add(&p,i,0,(added-count)*width))return 0;}
        }else if(s->kind==DSV41_C_COMPRESSED||s->kind==DSV41_C_INDEX){
            const uint64_t width=s->kind==DSV41_C_COMPRESSED?512u*4u:128u*4u;
            const uint64_t offset=(base/(s->index<3?2:1))*width;
            if(!dsv41_c_add(&p,i,offset,s->bytes-offset))return 0;
        }else if(!dsv41_c_add(&p,i,0,s->bytes))return 0;
    }
    p.artifact_bytes=204+(mode==3?60:52)+(uint64_t)added*4+129280*4+p.state_bytes;
    *out=p;return 1;
}
static int dsv41_c_delta(uint32_t context,uint32_t base,uint32_t target,dsv41_c_plan*out){
    return dsv41_c_delta_mode(3,context,base,target,out);
}
#endif
