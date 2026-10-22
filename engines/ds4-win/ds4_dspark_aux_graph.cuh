/* The same native drafter on the auxiliary GPU. Only parent embedding/head
 * activations and committed target hidden/KV rows cross the existing bounce. */
typedef struct {
    ds41_gpu_graph graph;
    ds41_draft draft;
    ds4_gpu_tensor *owned[512];
    uint32_t count;
} ds41_aux_graph;
static bool ds41_aux_own(ds41_aux_graph *a,ds4_gpu_tensor *tensor) {
    if(!tensor || a->count>=512)return false;
    a->owned[a->count++]=tensor;return true;
}
static void ds41_aux_discard(ds41_aux_graph *a) {
    if(!a)return;
    while(a->count)ds4_gpu_tensor_free(a->owned[--a->count]);
    free(a->graph.rows_view);
    free(a);
}
static ds41_aux_graph *ds41_aux_get(ds41_gpu_graph *parent) {
    ds41_draft *d=parent->draft;
    if(d->aux_graph)return (ds41_aux_graph *)d->aux_graph;
    ds41_aux_graph *a=xcalloc(1,sizeof(*a));a->graph=*parent;a->draft=*d;
    a->graph.draft=&a->draft;a->draft.aux_graph=NULL;a->draft.aux_parent=parent;
    ds41_gpu_graph *g=&a->graph;const uint32_t B=d->block;
    /* rows_view is a host pointer in the shallow graph copy: never replace
     * the target graph's view entries with auxiliary-device tensors. */
    g->rows_view=xcalloc(B,sizeof(*g->rows_view));
    bool ok=true;
#define DS41_AUX_BATCH(name,width) \
    if(ok){g->batch.name=ds4_gpu_dspark_aux_alloc((uint64_t)B*(width)*sizeof(float));ok=ds41_aux_own(a,g->batch.name);}
    DS41_PREFILL_ROWS(DS41_AUX_BATCH)
#undef DS41_AUX_BATCH
    for(uint32_t row=0;ok && row<B;row++) {
#define DS41_AUX_ROW(name,width) \
        if(ok){g->rows_view[row].name=ds4_gpu_tensor_view(g->batch.name,(uint64_t)row*(width)*sizeof(float),(uint64_t)(width)*sizeof(float));ok=ds41_aux_own(a,g->rows_view[row].name);}
        DS41_PREFILL_ROWS(DS41_AUX_ROW)
#undef DS41_AUX_ROW
    }
#define DS41_AUX_DRAFT(name) \
    if(ok){a->draft.name=ds4_gpu_dspark_aux_alloc(ds4_gpu_tensor_bytes(d->name));ok=ds41_aux_own(a,a->draft.name);}
    DS41_AUX_DRAFT(mh) DS41_AUX_DRAFT(pending) DS41_AUX_DRAFT(ring) DS41_AUX_DRAFT(raw)
    DS41_AUX_DRAFT(conf_proj) DS41_AUX_DRAFT(tokens) DS41_AUX_DRAFT(conf) DS41_AUX_DRAFT(parts)
    DS41_AUX_DRAFT(logits) DS41_AUX_DRAFT(x) DS41_AUX_DRAFT(xn) DS41_AUX_DRAFT(rows_logits)
    DS41_AUX_DRAFT(rows_x) DS41_AUX_DRAFT(rows_tmp)
#undef DS41_AUX_DRAFT
    if(ok){g->prefill_tokens=ds4_gpu_dspark_aux_alloc((uint64_t)B*sizeof(int32_t));ok=ds41_aux_own(a,g->prefill_tokens);}
    if(ok)ok=ds4_gpu_dspark_aux_copy(a->draft.conf_proj,d->conf_proj,ds4_gpu_tensor_bytes(d->conf_proj));
    if(!ok){ds41_aux_discard(a);return NULL;}
    d->aux_graph=a;
    fprintf(stderr,"ds4: whole auxiliary draft graph allocated; native stage/Markov paths reused, parent head/embedding on target\n");
    return a;
}
static void ds41_draft_aux_free(void *handle) {
    if(!handle)return;
    if(!ds4_gpu_dspark_aux_scope(1)){fprintf(stderr,"ds4: failed to enter auxiliary cleanup scope\n");abort();}
    ds41_aux_discard((ds41_aux_graph *)handle);
    if(!ds4_gpu_dspark_aux_scope(0)){fprintf(stderr,"ds4: failed auxiliary cleanup fence\n");abort();}
}
static bool ds41_draft_aux_pending(ds41_gpu_graph *parent) {
    if(!ds4_gpu_dspark_aux_scope(1))return false;
    ds41_aux_graph *a=ds41_aux_get(parent);bool ok=a!=NULL;
    if(ok) {
        ds41_draft *d=parent->draft;a->draft.mh_pos0=d->mh_pos0;a->draft.mh_rows=d->mh_rows;
        ok=ds4_gpu_dspark_aux_copy(a->draft.mh,d->mh,ds4_gpu_tensor_bytes(d->mh)) &&
            ds41_draft_pending_impl(&a->graph) &&
            ds4_gpu_dspark_aux_copy(d->pending,a->draft.pending,ds4_gpu_tensor_bytes(d->pending));
    }
    if(!ds4_gpu_dspark_aux_scope(0))ok=false;
    return ok;
}
static bool ds41_draft_aux_propose(ds41_gpu_graph *parent,const ds4_model *m,const ds4_weights *w,
        int token,int *tokens,float *confidence,bool public_confidence,bool *drained) {
    if(!ds4_gpu_dspark_aux_scope(1))return false;
    ds41_aux_graph *a=ds41_aux_get(parent);bool ok=a!=NULL;
    if(ok) {
        ds41_draft *d=parent->draft;a->graph.pos=parent->pos;
        a->draft.mh_pos0=d->mh_pos0;a->draft.mh_rows=d->mh_rows;
        ok=ds4_gpu_dspark_aux_copy(a->draft.ring,d->ring,ds4_gpu_tensor_bytes(d->ring)) &&
            ds41_draft_propose_impl(&a->graph,m,w,token,tokens,confidence,public_confidence,drained);
    }
    if(!ds4_gpu_dspark_aux_scope(0))ok=false;
    return ok;
}
