#ifndef QFN_NATIVE_LAYOUT_H
#define QFN_NATIVE_LAYOUT_H
/* QFN owner geometry shared by CUDA and Metal. This is a family adapter,
 * independent of the DSV artifact and its integrity contract. All offsets are
 * canonical native bytes: F16 KV/pool, F32 index and recurrent/history owners.
 * Mutable frontiers are replaced; immutable cache rows are appended. The last
 * predictor row may contain a provisional parent, so a delta replaces that row
 * too. No complete-state digest is required to derive or validate this plan. */
#include <stdint.h>
#include <stddef.h>

enum qfn_native_owner {
    QFN_GDN=1,QFN_CONV,QFN_KEY,QFN_VALUE,QFN_INDEX,QFN_POOL,
    QFN_PLE,QFN_PREV,QFN_ROPE,QFN_POS,QFN_CARRY
};
typedef struct {
    uint32_t layers,interval,embd,hc,head,kv_heads,vocab;
    uint32_t ssm_dim,ssm_heads,conv_channels,conv_history,index_dim,pool,ple_history;
    uint32_t context,mtp;
} qfn_native_shape;
typedef struct {uint32_t owner,layer;uint64_t offset,bytes;} qfn_native_slice;
#define QFN_NATIVE_SLICES 256u
typedef struct {
    uint32_t base,target,base_mtp,target_mtp,count;
    uint64_t bytes;
    qfn_native_slice slices[QFN_NATIVE_SLICES];
} qfn_native_plan;

static inline int qfn_native_add(qfn_native_plan *p,uint32_t owner,uint32_t layer,
                                 uint64_t offset,uint64_t bytes) {
    if(!bytes)return 1;
    if(p->count>=QFN_NATIVE_SLICES||bytes>UINT64_MAX-p->bytes)return 0;
    qfn_native_slice *r=&p->slices[p->count++];
    r->owner=owner;r->layer=layer;r->offset=offset;r->bytes=bytes;p->bytes+=bytes;return 1;
}
static inline int qfn_native_geometry(const qfn_native_shape *s,uint32_t base,uint32_t target,
        uint32_t base_mtp,uint32_t target_mtp,int include_carry,qfn_native_plan *p) {
    if(!s||!p||!target||base>=target||target>s->context||base_mtp>base||target_mtp>target||
       s->mtp>1||(!s->mtp&&(base_mtp||target_mtp))||!s->layers||s->layers>128||!s->interval||
       !s->embd||s->embd>65536||!s->hc||s->hc>64||!s->head||s->head>4096||
       !s->kv_heads||s->kv_heads>256||!s->vocab||s->vocab>1000000||
       !s->ssm_dim||s->ssm_dim>1024||!s->ssm_heads||s->ssm_heads>1024||
       !s->conv_channels||s->conv_channels>1000000||!s->conv_history||s->conv_history>64||
       !s->index_dim||s->index_dim>4096||s->pool!=4||!s->ple_history||s->ple_history>64)return 0;
    p->base=base;p->target=target;p->base_mtp=base_mtp;p->target_mtp=target_mtp;p->count=0;
    p->bytes=(uint64_t)(target-base)*4+(uint64_t)s->vocab*4;
    for(uint32_t l=0;l<s->layers+s->mtp;l++) {
        if(l<s->layers&&(l+1)%s->interval) {
            if(!qfn_native_add(p,QFN_GDN,l,0,(uint64_t)s->ssm_dim*s->ssm_dim*s->ssm_heads*4)||
               !qfn_native_add(p,QFN_CONV,l,0,(uint64_t)s->conv_channels*s->conv_history*4))return 0;
        } else {
            const uint32_t first=l==s->layers?(base_mtp?base_mtp-1:0):base;
            const uint32_t last=l==s->layers?target_mtp:target;
            if(last<first)return 0;
            const uint64_t kv=(uint64_t)s->head*s->kv_heads*2;
            if(!qfn_native_add(p,QFN_KEY,l,(uint64_t)first*kv,(uint64_t)(last-first)*kv)||
               !qfn_native_add(p,QFN_VALUE,l,(uint64_t)first*kv,(uint64_t)(last-first)*kv)||
               !qfn_native_add(p,QFN_INDEX,l,(uint64_t)first*s->index_dim*4,(uint64_t)(last-first)*s->index_dim*4)||
               !qfn_native_add(p,QFN_POOL,l,(uint64_t)(first/s->pool)*s->index_dim*2,
                   (uint64_t)(last/s->pool-first/s->pool)*s->index_dim*2))return 0;
        }
    }
    return qfn_native_add(p,QFN_PLE,0,0,(uint64_t)s->ple_history*s->embd*s->hc*4)&&
        qfn_native_add(p,QFN_PREV,0,0,32)&&qfn_native_add(p,QFN_ROPE,0,0,4)&&
        qfn_native_add(p,QFN_POS,0,(uint64_t)base*16,(uint64_t)(target-base)*16)&&
        (!include_carry||qfn_native_add(p,QFN_CARRY,0,(uint64_t)base*s->embd*s->hc*4,
            (uint64_t)(target-base)*s->embd*s->hc*4));
}
/* A rewind anchor owns mutable histories and the potentially overwritten
 * predictor frontier. Prefix KV/index/position arrays remain in their native
 * owners; saving an anchor never copies the full context. */
static inline int qfn_native_anchor_geometry(const qfn_native_shape *s,uint32_t position,
        uint32_t mtp_rows,qfn_native_plan *p) {
    qfn_native_plan validation;
    if(!qfn_native_geometry(s,0,position,0,mtp_rows,0,&validation))return 0;
    p->base=p->target=position;p->base_mtp=p->target_mtp=mtp_rows;p->count=0;p->bytes=0;
    for(uint32_t i=0;i<validation.count;i++) {
        const qfn_native_slice *r=&validation.slices[i];
        if(r->owner==QFN_GDN||r->owner==QFN_CONV||r->owner==QFN_PLE||r->owner==QFN_PREV||r->owner==QFN_ROPE) {
            if(!qfn_native_add(p,r->owner,r->layer,r->offset,r->bytes))return 0;
        } else if(r->layer==s->layers&&mtp_rows&&r->owner>=QFN_KEY&&r->owner<=QFN_POOL) {
            const uint64_t stride=r->owner==QFN_INDEX?(uint64_t)s->index_dim*4:
                r->owner==QFN_POOL?(uint64_t)s->index_dim*2:(uint64_t)s->head*s->kv_heads*2;
            const uint32_t rows=r->owner==QFN_POOL?mtp_rows/s->pool:mtp_rows;
            if(rows&&!qfn_native_add(p,r->owner,r->layer,(uint64_t)(rows-1)*stride,stride))return 0;
        }
    }
    return 1;
}
#endif
