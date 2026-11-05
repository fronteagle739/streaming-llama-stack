/* QFN family owner adapter. Called under the existing generation owner, just
 * like ds4_session_native_apply_delta. No DSV layout/contract is modified. */
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
int ds4_qfn_session_shape(ds4_session *s,qfn_native_shape *shape,uint32_t *mtp,char *err,size_t cap) {
    DS4_POOL_SCOPE();
    if(!s||!shape||!mtp||!ds4_engine_is_qwen4(s->engine)||!s->qwen4_graph_ready||
       !s->checkpoint_valid||s->qwen4_rewound||s->qwen4_graph.pos!=(uint32_t)s->checkpoint.len||
       !ds4_gpu_synchronize())return payload_set_err(err,cap,"QFN owner frontier is incomplete"),1;
    *shape=(qfn_native_shape){DS4_N_LAYER-DS4_N_NEXTN_PREDICT,4,DS4_N_EMBD,DS4_N_HC,
        DS4_N_HEAD_DIM,DS4_N_HEAD_KV,DS4_N_VOCAB,DS4_N_LIN_HEAD_DIM,DS4_N_LIN_V_HEAD,
        DS4_N_LIN_CONV_DIM,DS4_N_LIN_CONV-1,DS4_N_INDEXER_HEAD_DIM,4,
        (DS4_N_PLE_CONV-1)*DS4_N_PLE_NGRAM,(uint32_t)s->ctx_size,DS4_N_NEXTN_PREDICT};
    *mtp=s->qwen4_graph.mtp_pos< s->qwen4_graph.pos?s->qwen4_graph.mtp_pos:s->qwen4_graph.pos;return 0;
}
int ds4_qfn_session_pair(ds4_session *s,char *err,size_t cap) {
    DS4_POOL_SCOPE();
    qfn_native_shape shape;uint32_t mtp;
    if(ds4_qfn_session_shape(s,&shape,&mtp,err,cap))return 1;
    ds4_qwen4_gpu_graph *g=&s->qwen4_graph;
    if(!g->paired_carry)g->paired_carry=ds4_gpu_tensor_alloc((uint64_t)shape.context*shape.embd*shape.hc*4);
    if(!g->paired_carry)return payload_set_err(err,cap,"QFN paired residual history allocation failed under host budget"),1;
    g->paired_carry_from=g->pos;return 0;
}
static int qfn_native_owner_io(ds4_session *s,const qfn_native_slice *r,void *data,bool write) {
    ds4_qwen4_gpu_graph *g=&s->qwen4_graph;ds4_gpu_tensor *t=NULL;
    if(!data||!r->bytes||r->bytes>65536||r->layer>=DS4_N_LAYER)return 0;
    switch(r->owner) {
        case QFN_GDN:t=g->layer_lin_state[r->layer];break;
        case QFN_CONV:t=g->layer_lin_hist[r->layer];break;
        case QFN_KEY:t=g->layer_k_cache[r->layer];break;
        case QFN_VALUE:t=g->layer_v_cache[r->layer];break;
        case QFN_INDEX:t=g->layer_ik_cache[r->layer];break;
        case QFN_POOL:t=g->layer_block_key[r->layer];break;
        case QFN_PLE:t=g->ple_hist;break;
        case QFN_POS:t=g->pos3;break;
        case QFN_CARRY:
            if(r->offset<(uint64_t)g->paired_carry_from*DS4_N_EMBD*DS4_N_HC*4)return 0;
            t=g->paired_carry;break;
        case QFN_PREV:
            if(r->offset||r->bytes!=32)return 0;
            if(write)memcpy(g->ple_prev,data,32);else memcpy(data,g->ple_prev,32);return 1;
        case QFN_ROPE:
            if(r->offset||r->bytes!=4)return 0;
            if(write)memcpy(&g->mrope_delta,data,4);else memcpy(data,&g->mrope_delta,4);return 1;
        default:return 0;
    }
    return t&&(write?ds4_gpu_tensor_write(t,r->offset,data,r->bytes):ds4_gpu_tensor_read(t,r->offset,data,r->bytes))!=0;
}
static int qfn_native_values(const qfn_native_slice *r,const void *data) {
    if(r->owner==QFN_GDN||r->owner==QFN_CONV||r->owner==QFN_INDEX||r->owner==QFN_PLE||r->owner==QFN_CARRY) {
        if(r->bytes%4)return 0;
        for(uint64_t i=0;i<r->bytes;i+=4){float f;memcpy(&f,(const char*)data+i,4);if(!isfinite(f))return 0;}
    } else if(r->owner==QFN_KEY||r->owner==QFN_VALUE||r->owner==QFN_POOL) {
        if(r->bytes%2)return 0;
        for(uint64_t i=0;i<r->bytes;i+=2){uint16_t h;memcpy(&h,(const char*)data+i,2);if((h&0x7c00)==0x7c00)return 0;}
    } else if(r->owner==QFN_PREV) {
        for(uint64_t i=0;i<r->bytes;i+=4){uint32_t t;memcpy(&t,(const char*)data+i,4);if(t>=DS4_N_VOCAB)return 0;}
    }
    return 1;
}
struct ds4_qfn_anchor {
    qfn_native_plan plan;qfn_native_shape shape;
    token_vec tokens;float *logits;unsigned char *owners;
    ds4_vision_identity *images;size_t image_count;
    uint64_t cache_epoch;
};
/* Opt-in test observation of the actual immutable mutable-owner anchor.
 * No owner hash, extra state transfer or file exists in the production route. */
uint64_t ds4_qfn_anchor_test_bytes(const ds4_qfn_anchor *a) {
    return a?24u+(uint64_t)a->plan.count*24u+a->plan.bytes+(uint64_t)a->shape.vocab*4u+(uint64_t)a->tokens.len*4u:0;
}
int ds4_qfn_anchor_test_write(const ds4_qfn_anchor *a,ds4_native_write_fn write,void *stream) {
    if(!a||!write)return 0;
    const uint32_t h[6]={0x4f4e4651,1,a->shape.context,(uint32_t)a->tokens.len,a->shape.vocab,a->plan.count};
    if(!write(stream,h,sizeof(h)))return 0;
    for(uint32_t i=0;i<a->plan.count;i++) {
        const qfn_native_slice *p=&a->plan.slices[i];uint32_t kind[2]={p->owner,p->layer};uint64_t extent[2]={p->offset,p->bytes};
        if(!write(stream,kind,sizeof(kind))||!write(stream,extent,sizeof(extent)))return 0;
    }
    const void *parts[3]={a->owners,a->logits,a->tokens.v};
    const uint64_t sizes[3]={a->plan.bytes,(uint64_t)a->shape.vocab*4u,(uint64_t)a->tokens.len*4u};
    for(int i=0;i<3;i++)for(uint64_t at=0;at<sizes[i];) {
        size_t n=(size_t)(sizes[i]-at);if(n>65536)n=65536;
        if(!write(stream,(const char*)parts[i]+at,n))return 0;at+=n;
    }
    return 1;
}
void ds4_qfn_anchor_free(ds4_qfn_anchor *a) {
    if(!a)return;token_vec_free(&a->tokens);free(a->logits);free(a->owners);free(a->images);free(a);
}
int ds4_qfn_anchor_save(ds4_session *s,ds4_qfn_anchor **out,uint64_t budget,uint64_t *bytes,char *err,size_t cap) {
    DS4_POOL_SCOPE();*bytes=0;qfn_native_shape shape;uint32_t mtp;
    if(!out||ds4_qfn_session_shape(s,&shape,&mtp,err,cap))return 1;
    ds4_qfn_anchor *a=calloc(1,sizeof(*a));if(!a)return 1;a->shape=shape;a->cache_epoch=s->qwen4_graph.paired_cache_epoch;
    if(!qfn_native_anchor_geometry(&shape,s->qwen4_graph.pos,mtp,&a->plan)||a->plan.bytes>SIZE_MAX||
       a->plan.bytes+(uint64_t)shape.vocab*4+(uint64_t)s->checkpoint.len*4+
           s->checkpoint_image_count*sizeof(ds4_vision_identity)>budget)goto failed;
    a->owners=malloc((size_t)a->plan.bytes);a->logits=malloc((size_t)shape.vocab*4);
    if(!a->owners||!a->logits)goto failed;
    for(int i=0;i<s->checkpoint.len;i++)token_vec_push(&a->tokens,s->checkpoint.v[i]);
    memcpy(a->logits,s->logits,(size_t)shape.vocab*4);a->image_count=s->checkpoint_image_count;
    if(a->image_count){a->images=malloc(a->image_count*sizeof(*a->images));if(!a->images)goto failed;
        memcpy(a->images,s->checkpoint_images,a->image_count*sizeof(*a->images));}
    size_t cursor=0;
    for(uint32_t i=0;i<a->plan.count;i++)for(uint64_t off=0;off<a->plan.slices[i].bytes;) {
        const qfn_native_slice *r=&a->plan.slices[i];size_t n=(size_t)(r->bytes-off);if(n>65536)n=65536;
        const qfn_native_slice part={r->owner,r->layer,r->offset+off,n};
        if(!qfn_native_owner_io(s,&part,a->owners+cursor,false))goto failed;off+=n;cursor+=n;
    }
    if(cursor!=a->plan.bytes)goto failed;
    *bytes=a->plan.bytes+(uint64_t)shape.vocab*4+(uint64_t)a->tokens.len*4+a->image_count*sizeof(*a->images);
    ds4_qfn_anchor_free(*out);*out=a;return 0;
failed:ds4_qfn_anchor_free(a);payload_set_err(err,cap,"QFN mutable rewind anchor capture failed or exceeded budget");return 1;
}
int ds4_qfn_anchor_restore(ds4_session *s,const ds4_qfn_anchor *a,char *err,size_t cap) {
    DS4_POOL_SCOPE();
    if(!s||!a||!a->owners||!s->qwen4_graph_ready||!ds4_engine_is_qwen4(s->engine)||
       (uint32_t)s->ctx_size!=a->shape.context||s->qwen4_graph.paired_cache_epoch!=a->cache_epoch||
       !ds4_gpu_synchronize())return payload_set_err(err,cap,"QFN rewind anchor family/context/cache epoch differs"),1;
    s->checkpoint_valid=false;ds4_qwen4_gpu_graph *g=&s->qwen4_graph;size_t cursor=0;
    for(uint32_t i=0;i<a->plan.count;i++)for(uint64_t off=0;off<a->plan.slices[i].bytes;) {
        const qfn_native_slice *r=&a->plan.slices[i];size_t n=(size_t)(r->bytes-off);if(n>65536)n=65536;
        const qfn_native_slice part={r->owner,r->layer,r->offset+off,n};
        if(!qfn_native_owner_io(s,&part,a->owners+cursor,true))goto failed;off+=n;cursor+=n;
    }
    if(cursor!=a->plan.bytes||!ds4_gpu_synchronize())goto failed;
    token_vec next={0};for(int i=0;i<a->tokens.len;i++)token_vec_push(&next,a->tokens.v[i]);
    ds4_vision_identity *images=a->image_count?malloc(a->image_count*sizeof(*images)):NULL;
    if(a->image_count&&!images){token_vec_free(&next);goto failed;}
    if(images)memcpy(images,a->images,a->image_count*sizeof(*images));
    token_vec_free(&s->checkpoint);s->checkpoint=next;memcpy(s->logits,a->logits,(size_t)a->shape.vocab*4);
    free(s->checkpoint_images);s->checkpoint_images=images;s->checkpoint_image_count=a->image_count;
    g->pos=a->plan.target;g->mtp_pos=a->plan.target_mtp;g->paired_carry_from=g->pos;
    g->snap_valid=g->snap2_valid=g->snap0_valid=false;g->snap_after_first=g->snap_after_second=false;
    s->qwen4_rewound=false;s->mtp_draft_valid=false;s->glm_mtp_have=0;s->glm_mtp_have2=false;s->checkpoint_valid=true;return 0;
failed:ds4_session_invalidate(s);payload_set_err(err,cap,"QFN mutable rewind restore failed; checkpoint invalid");return 1;
}
int ds4_qfn_anchor_check(ds4_session *s,const ds4_qfn_anchor *a,uint64_t *bytes,char *err,size_t cap) {
    DS4_POOL_SCOPE();*bytes=0;qfn_native_shape shape;uint32_t mtp;
    if(!a||ds4_qfn_session_shape(s,&shape,&mtp,err,cap)||s->qwen4_graph.pos!=a->plan.target||mtp!=a->plan.target_mtp||
       s->qwen4_graph.paired_cache_epoch!=a->cache_epoch||s->checkpoint.len!=a->tokens.len||
       memcmp(s->checkpoint.v,a->tokens.v,(size_t)a->tokens.len*4)||
       memcmp(s->logits,a->logits,(size_t)shape.vocab*4)||s->checkpoint_image_count!=a->image_count||
       (a->image_count&&memcmp(s->checkpoint_images,a->images,a->image_count*sizeof(*a->images))))goto failed;
    unsigned char page[65536];uint64_t cursor=0;
    for(uint32_t i=0;i<a->plan.count;i++)for(uint64_t off=0;off<a->plan.slices[i].bytes;) {
        const qfn_native_slice *r=&a->plan.slices[i];size_t n=(size_t)(r->bytes-off);if(n>sizeof(page))n=sizeof(page);
        const qfn_native_slice part={r->owner,r->layer,r->offset+off,n};
        if(!qfn_native_owner_io(s,&part,page,false)||memcmp(page,a->owners+cursor,n))goto failed;
        off+=n;cursor+=n;
    }
    if(cursor!=a->plan.bytes)goto failed;*bytes=cursor+(uint64_t)shape.vocab*4+(uint64_t)a->tokens.len*4;
    return 0;
failed:ds4_session_invalidate(s);payload_set_err(err,cap,"QFN test actual mutable-owner readback differs from saved anchor");return 1;
}
int ds4_qfn_session_delta_write(ds4_session *s,uint32_t base,uint32_t base_mtp,uint64_t budget,
        ds4_native_write_fn write,void *stream,uint64_t *bytes,char *err,size_t cap) {
    DS4_POOL_SCOPE();
    qfn_native_shape shape;uint32_t mtp;qfn_native_plan plan;*bytes=0;
    if(!write||ds4_qfn_session_shape(s,&shape,&mtp,err,cap)||
       !qfn_native_geometry(&shape,base,s->qwen4_graph.pos,base_mtp,mtp,1,&plan)||
       plan.bytes>UINT64_MAX-40||plan.bytes+40>budget)return payload_set_err(err,cap,"QFN reverse delta geometry/budget refused"),1;
    const uint64_t extent=plan.bytes+40;
    const uint32_t h[10]={0x444e4651,1,shape.context,base,plan.target,base_mtp,mtp,1,(uint32_t)extent,(uint32_t)(extent>>32)};
    if(!write(stream,h,sizeof(h)))goto failed;
    const unsigned char *ids=(const unsigned char*)(s->checkpoint.v+base);
    for(uint64_t at=0,n=(uint64_t)(plan.target-base)*4;at<n;) {
        size_t k=(size_t)(n-at);if(k>65536)k=65536;if(!write(stream,ids+at,k))goto failed;at+=k;
    }
    for(uint64_t at=0,n=(uint64_t)shape.vocab*4;at<n;) {
        size_t k=(size_t)(n-at);if(k>65536)k=65536;if(!write(stream,(char*)s->logits+at,k))goto failed;at+=k;
    }
    unsigned char page[65536];
    for(uint32_t i=0;i<plan.count;i++)for(uint64_t at=0;at<plan.slices[i].bytes;) {
        const qfn_native_slice *r=&plan.slices[i];size_t n=(size_t)(r->bytes-at);if(n>sizeof(page))n=sizeof(page);
        qfn_native_slice part={r->owner,r->layer,r->offset+at,n};
        if((s->cancel&&s->cancel(s->cancel_ud))||!qfn_native_owner_io(s,&part,page,false)||!write(stream,page,n))goto failed;
        at+=n;
    }
    *bytes=extent;return 0;
failed:payload_set_err(err,cap,"QFN reverse owner stream failed");return 1;
}
int ds4_qfn_session_delta_apply(ds4_session *s,uint32_t base,uint32_t target,uint32_t base_mtp,uint32_t target_mtp,
        uint64_t bytes,uint64_t budget,ds4_native_read_fn read,void *stream,char *err,size_t cap) {
    DS4_POOL_SCOPE();
    qfn_native_shape shape;uint32_t mtp;qfn_native_plan plan;bool mutated=false;int rc=1;
    token_vec next={0};float *logits=NULL;
    if(!read||ds4_qfn_session_shape(s,&shape,&mtp,err,cap)||s->qwen4_graph.pos!=base||mtp!=base_mtp||
       !qfn_native_geometry(&shape,base,target,base_mtp,target_mtp,0,&plan)||
       plan.bytes>UINT64_MAX-40||bytes!=plan.bytes+40||bytes>budget)
        return payload_set_err(err,cap,"QFN forward delta base/extent/budget refused"),1;
    uint32_t h[10];
    if(!read(stream,h,sizeof(h))||h[0]!=0x444e4651||h[1]!=1||!h[2]||h[2]<target||h[3]!=base||h[4]!=target||
       h[5]!=base_mtp||h[6]!=target_mtp||h[7]||(((uint64_t)h[9]<<32)|h[8])!=bytes)goto done;
    for(uint32_t i=0;i<base;i++)token_vec_push(&next,s->checkpoint.v[i]);
    for(uint32_t i=base;i<target;i++){uint32_t t;if(!read(stream,&t,4)||t>=shape.vocab)goto done;token_vec_push(&next,(int)t);}
    logits=malloc((size_t)shape.vocab*4);if(!logits)goto done;
    for(uint64_t at=0,n=(uint64_t)shape.vocab*4;at<n;) {
        size_t k=(size_t)(n-at);if(k>65536)k=65536;if(!read(stream,(char*)logits+at,k))goto done;at+=k;
    }
    for(uint32_t i=0;i<shape.vocab;i++)if(!isfinite(logits[i]))goto done;
    mutated=true;s->checkpoint_valid=false;s->mtp_draft_valid=false;s->glm_mtp_have=0;s->glm_mtp_have2=false;
    ds4_qwen4_gpu_graph *g=&s->qwen4_graph;g->snap_valid=g->snap2_valid=g->snap0_valid=false;
    unsigned char page[65536];
    for(uint32_t i=0;i<plan.count;i++)for(uint64_t at=0;at<plan.slices[i].bytes;) {
        const qfn_native_slice *r=&plan.slices[i];size_t n=(size_t)(r->bytes-at);if(n>sizeof(page))n=sizeof(page);
        qfn_native_slice part={r->owner,r->layer,r->offset+at,n};
        if((s->cancel&&s->cancel(s->cancel_ud))||!read(stream,page,n)||!qfn_native_values(&part,page)||
           !qfn_native_owner_io(s,&part,page,true))goto done;at+=n;
    }
    for(uint32_t i=0;i<8;i++)if(g->ple_prev[i]!=(i<target?next.v[target-1-i]:DS4_PLE_EOS_ID))goto done;
    if(!ds4_gpu_synchronize())goto done;
    token_vec_free(&s->checkpoint);s->checkpoint=next;memset(&next,0,sizeof(next));
    memcpy(s->logits,logits,(size_t)shape.vocab*4);g->pos=target;g->mtp_pos=target_mtp;g->paired_carry_from=target;
    s->qwen4_rewound=false;s->checkpoint_valid=true;rc=0;
done:
    free(logits);token_vec_free(&next);if(rc){if(mutated)ds4_session_invalidate(s);payload_set_err(err,cap,"QFN forward delta failed; partial owners invalidated");}return rc;
}
#else
uint64_t ds4_qfn_anchor_test_bytes(const ds4_qfn_anchor*a){(void)a;return 0;}
int ds4_qfn_anchor_test_write(const ds4_qfn_anchor*a,ds4_native_write_fn w,void*u){(void)a;(void)w;(void)u;return 0;}
void ds4_qfn_anchor_free(ds4_qfn_anchor*a){(void)a;}
int ds4_qfn_anchor_save(ds4_session*s,ds4_qfn_anchor**a,uint64_t b,uint64_t*x,char*e,size_t n){(void)s;(void)a;(void)b;(void)x;payload_set_err(e,n,"QFN paired anchors require Metal");return 1;}
int ds4_qfn_anchor_restore(ds4_session*s,const ds4_qfn_anchor*a,char*e,size_t n){(void)s;(void)a;payload_set_err(e,n,"QFN paired anchors require Metal");return 1;}
int ds4_qfn_anchor_check(ds4_session*s,const ds4_qfn_anchor*a,uint64_t*b,char*e,size_t n){(void)s;(void)a;(void)b;payload_set_err(e,n,"QFN paired anchor readback requires Metal");return 1;}
int ds4_qfn_session_pair(ds4_session *s,char *e,size_t n){(void)s;payload_set_err(e,n,"QFN paired owners require Metal");return 1;}
int ds4_qfn_session_shape(ds4_session *s,qfn_native_shape *g,uint32_t *m,char *e,size_t n){(void)s;(void)g;(void)m;payload_set_err(e,n,"QFN paired owners require Metal");return 1;}
int ds4_qfn_session_delta_write(ds4_session*s,uint32_t b,uint32_t m,uint64_t z,ds4_native_write_fn w,void*u,uint64_t*x,char*e,size_t n){(void)s;(void)b;(void)m;(void)z;(void)w;(void)u;(void)x;payload_set_err(e,n,"QFN paired owners require Metal");return 1;}
int ds4_qfn_session_delta_apply(ds4_session*s,uint32_t b,uint32_t t,uint32_t m,uint32_t q,uint64_t z,uint64_t l,ds4_native_read_fn r,void*u,char*e,size_t n){(void)s;(void)b;(void)t;(void)m;(void)q;(void)z;(void)l;(void)r;(void)u;payload_set_err(e,n,"QFN paired owners require Metal");return 1;}
#endif
