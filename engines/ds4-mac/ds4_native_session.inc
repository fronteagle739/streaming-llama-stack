#include "ds4_native_session.h"
#include "dsv41-native-c-artifact.h"

#if defined(__APPLE__) && !defined(DS4_NO_GPU)
typedef struct {
    ds41_state_span spans[55];
    dsv41_c_span geometry[55];
    uint32_t count;
    ds4_session *session;
    bool active;
} ds4_native_lease;

static int ds4_native_read_owner(void *ud,uint32_t index,uint64_t offset,void *dst,size_t bytes){
    ds4_native_lease*l=ud;
    if(!l->active||!dst||!bytes||bytes>65536||index>=l->count||
       offset>l->spans[index].bytes||bytes>l->spans[index].bytes-offset||
       (l->session->cancel&&l->session->cancel(l->session->cancel_ud)))return 0;
    return ds4_gpu_tensor_read(l->spans[index].tensor,offset,dst,bytes)!=0;
}
static int ds4_native_ready(ds4_session*s,char*err,size_t cap){
    if(!s||!s->engine||s->engine->backend!=DS4_BACKEND_METAL||s->distributed||s->engine->tp.active||
       !ds4_session_is_ds41(s)||!s->ds41_graph_ready||!s->checkpoint_valid){
        payload_set_err(err,cap,"native delta requires a complete local Metal V4.1 session");return 0;}
    ds41_gpu_graph*g=&s->ds41_graph;
    if(!g->valid||!g->pos||g->pos!=(uint32_t)s->checkpoint.len||!g->dspark_capture||!g->dspark_hidden||
       !g_ds41_levers.mtp_state_fix||g->dspark_hidden_end!=g->pos||
       g->dspark_hidden_len!=(g->pos<128?g->pos:128)||!ds4_gpu_synchronize()){
        payload_set_err(err,cap,"native delta requires exact mode3 DSpark frontier");return 0;}
    return 1;
}
static int ds4_native_visit_at(ds4_session*s,uint32_t position,const int32_t*tokens,const float*logits,
        ds4_native_visit_fn visitor,void*ud,char*err,size_t cap){
    ds4_native_lease lease={0};lease.session=s;
    if(!visitor||!dsv41_c_layout(s->ds41_graph.ctx,position,lease.geometry,&lease.count)){
        payload_set_err(err,cap,"native delta invalid view geometry or verifier");return 1;}
    uint32_t n=ds41_state_spans(&s->ds41_graph,position,lease.spans);
    if(n!=lease.count){payload_set_err(err,cap,"native delta owner count differs");return 1;}
    for(uint32_t i=0;i<n;++i)if(lease.spans[i].bytes!=lease.geometry[i].bytes){
        payload_set_err(err,cap,"native delta owner extent differs");return 1;}
    ds4_native_view view={s->ds41_graph.ctx,position,position,position<128?position:128,
        n,tokens,logits,lease.geometry,ds4_native_read_owner,&lease};
    lease.active=true;int ok=visitor(ud,&view,err,cap);lease.active=false;
    if(!ok&&err&&cap&&!err[0])payload_set_err(err,cap,"native delta identity verification failed");
    return ok?0:1;
}
int ds4_session_native_visit(ds4_session*s,ds4_native_visit_fn visitor,void*ud,char*err,size_t cap){
    if(!ds4_native_ready(s,err,cap))return 1;
    return ds4_native_visit_at(s,s->ds41_graph.pos,(const int32_t*)s->checkpoint.v,s->logits,visitor,ud,err,cap);
}
int ds4_session_native_apply_delta(ds4_session*s,uint32_t base,uint32_t target,
        const int32_t*new_tokens,const float*logits,ds4_native_read_fn read,void*stream,
        ds4_native_visit_fn verify_base,void*base_identity,
        ds4_native_visit_fn verify_target,void*target_identity,char*err,size_t cap){
    dsv41_c_plan plan;
    if(!ds4_native_ready(s,err,cap))return 1;
    if(s->ds41_graph.pos!=base||!read||!verify_base||!verify_target||
       !dsv41_c_delta(s->ds41_graph.ctx,base,target,&plan)||
       !dsv41_c_values_valid(new_tokens,target-base,logits)){
        payload_set_err(err,cap,"native delta base, geometry or values invalid");return 1;}
    if(ds4_session_native_visit(s,verify_base,base_identity,err,cap))return 1;
    ds4_tokens tokens={0};tokens.v=malloc((size_t)target*sizeof(int));
    if(!tokens.v){payload_set_err(err,cap,"native delta token allocation failed");return 1;}
    tokens.cap=tokens.len=(int)target;
    memcpy(tokens.v,s->checkpoint.v,(size_t)base*4);
    memcpy(tokens.v+base,new_tokens,(size_t)(target-base)*4);
    ds41_gpu_graph*g=&s->ds41_graph;ds41_state_span spans[55];
    uint32_t n=ds41_state_spans(g,target,spans);int rc=1;
    if(n!=plan.span_count)goto done;
    for(uint32_t i=0;i<n;++i)if(spans[i].bytes!=plan.spans[i].bytes)goto done;
    s->checkpoint_valid=false;g->valid=false;s->mtp_draft_valid=false;
    ds4_session_dspark_capture_invalidate(s);
    memcpy(s->logits,logits,129280*4);
    uint8_t buffer[65536];
    for(uint32_t i=0;i<plan.slice_count;++i){const dsv41_c_slice*r=plan.slices+i;
        for(uint64_t off=0;off<r->bytes;){size_t bytes=(size_t)(r->bytes-off<sizeof(buffer)?r->bytes-off:sizeof(buffer));
            if((s->cancel&&s->cancel(s->cancel_ud))||!read(stream,buffer,bytes)||
               !ds4_gpu_tensor_write(spans[r->span].tensor,r->offset+off,buffer,bytes))goto done;
            off+=bytes;}}
    if(!ds4_gpu_synchronize()||ds4_native_visit_at(s,target,(const int32_t*)tokens.v,s->logits,
       verify_target,target_identity,err,cap))goto done;
    /* Same commit sequence as ds41_load_payload; generation changes invalidate
     * drafter reuse, and the next ordinary decode warms it from this ring. */
    ds41_graph_reset(g);
    for(uint32_t i=0;i<3&&i<target;++i)g->history.tail[i]=(int32_t)g->token_map[tokens.v[target-1-i]];
    g->pos=target;g->dspark_hidden_end=target;g->dspark_hidden_len=target<128?target:128;
    ds4_tokens_free(&s->checkpoint);s->checkpoint=tokens;memset(&tokens,0,sizeof(tokens));
    s->checkpoint_valid=true;rc=0;
done:
    ds4_tokens_free(&tokens);
    if(rc&&err&&cap&&!err[0])payload_set_err(err,cap,"native delta apply failed; partial checkpoint remains invalid");
    return rc;
}
int ds4_session_native_rewind(ds4_session*s,uint32_t target,const float*logits,
        ds4_native_read_fn read,void*stream,ds4_native_visit_fn verify_base,void*base_identity,
        ds4_native_visit_fn verify_target,void*target_identity,char*err,size_t cap){
    if(!ds4_native_ready(s,err,cap))return 1;
    if(!target||target>=s->ds41_graph.pos||!logits||!read||!verify_base||!verify_target||
       !dsv41_c_values_valid((const int32_t*)s->checkpoint.v,target,logits))
        return payload_set_err(err,cap,"native rewind target or callbacks invalid"),1;
    if(ds4_session_native_visit(s,verify_base,base_identity,err,cap))return 1;
    ds41_gpu_graph*g=&s->ds41_graph;ds41_state_span spans[55];dsv41_c_span geometry[55];uint32_t count=0;
    if(!dsv41_c_layout(g->ctx,target,geometry,&count)||ds41_state_spans(g,target,spans)!=count)return 1;
    /* Existing rewind invalidates recurrent state. Restore the saved mutable
     * frontier before publishing, using the same commit as native import. */
    ds4_session_rewind(s,(int)target);g->valid=false;
    uint8_t page[65536];
    for(uint32_t i=0;i<count;i++){
        if(spans[i].bytes!=geometry[i].bytes)goto failed;
        if(geometry[i].kind==DSV41_C_COMPRESSED||geometry[i].kind==DSV41_C_INDEX)continue;
        for(uint64_t off=0;off<spans[i].bytes;){size_t n=(size_t)(spans[i].bytes-off);if(n>sizeof(page))n=sizeof(page);
            if((s->cancel&&s->cancel(s->cancel_ud))||!read(stream,page,n)||!ds4_gpu_tensor_write(spans[i].tensor,off,page,n))goto failed;
            off+=n;
        }
    }
    memcpy(s->logits,logits,129280*4);
    if(!ds4_gpu_synchronize()||ds4_native_visit_at(s,target,(const int32_t*)s->checkpoint.v,s->logits,
       verify_target,target_identity,err,cap))goto failed;
    ds41_graph_reset(g);
    for(uint32_t i=0;i<3&&i<target;++i)g->history.tail[i]=(int32_t)g->token_map[s->checkpoint.v[target-1-i]];
    g->pos=target;g->dspark_hidden_end=target;g->dspark_hidden_len=target<128?target:128;
    s->checkpoint_valid=true;return 0;
failed:
    ds4_session_invalidate(s);payload_set_err(err,cap,"native rewind restore failed; checkpoint invalid");return 1;
}
#else
int ds4_session_native_visit(ds4_session*s,ds4_native_visit_fn fn,void*ud,char*err,size_t cap){
    (void)s;(void)fn;(void)ud;payload_set_err(err,cap,"native delta requires Metal V4.1");return 1;
}
int ds4_session_native_apply_delta(ds4_session*s,uint32_t b,uint32_t t,const int32_t*ids,const float*l,
        ds4_native_read_fn r,void*stream,ds4_native_visit_fn vb,void*bi,ds4_native_visit_fn vt,void*ti,char*err,size_t cap){
    (void)s;(void)b;(void)t;(void)ids;(void)l;(void)r;(void)stream;(void)vb;(void)bi;(void)vt;(void)ti;
    payload_set_err(err,cap,"native delta requires Metal V4.1");return 1;
}
int ds4_session_native_rewind(ds4_session*s,uint32_t t,const float*l,ds4_native_read_fn r,void*stream,
        ds4_native_visit_fn vb,void*bi,ds4_native_visit_fn vt,void*ti,char*err,size_t cap){
    (void)s;(void)t;(void)l;(void)r;(void)stream;(void)vb;(void)bi;(void)vt;(void)ti;
    payload_set_err(err,cap,"native rewind requires Metal V4.1");return 1;
}
#endif
