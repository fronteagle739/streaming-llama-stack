/* CPU-only independent owner bytes: immutable compressed prefixes and mutable
 * ring/tail owners. Compare an incremental catch-up snapshot with a full hash. */
typedef struct {dsv41_c_span spans[55];uint32_t count,position;bool corrupt;} async_fixture;
static int async_fixture_read(void*ud,uint32_t index,uint64_t off,void*dst,size_t n){
    async_fixture*f=ud;if(index>=f->count||off+n>f->spans[index].bytes)return 0;
    const dsv41_c_span*s=&f->spans[index];bool append=s->kind==DSV41_C_COMPRESSED||s->kind==DSV41_C_INDEX;
    for(size_t i=0;i<n;i++)((unsigned char*)dst)[i]=(unsigned char)((off+i)*17+s->index*31+s->kind*7+(append?0:f->position));
    if(f->corrupt&&append&&off==0&&n)((unsigned char*)dst)[0]^=1;
    return 1;
}
static ds4_native_view async_fixture_view(async_fixture*f,uint32_t pos,int32_t*tokens,float*logits){
    memset(f,0,sizeof(*f));f->position=pos;assert(dsv41_c_layout(1000001,pos,f->spans,&f->count));
    ds4_native_view v={1000001,pos,pos,pos<128?pos:128,f->count,tokens,logits,f->spans,async_fixture_read,f};return v;
}
static void test_async_snapshots(void){
    int32_t tokens[256];for(int i=0;i<256;i++)tokens[i]=i+10;
    float*logits=calloc(129280,sizeof(float));assert(logits);char err[256]={0};
    async_fixture base_fixture,target_fixture;ds4_native_view base=async_fixture_view(&base_fixture,40,tokens,logits);
    ds4_ns_session s={0};assert(ds4_native_capture(&s.synced,&base,err,sizeof(err)));
    assert(ds4_ns_chain_visit(&s,&base,err,sizeof(err)));
    for(uint32_t pos=64;pos<=65;pos++){
        ds4_native_view target=async_fixture_view(&target_fixture,pos,tokens,logits);
        ds4_ns_snap snapshot={&s,NULL,0};assert(ds4_ns_snapshot_visit(&snapshot,&target,err,sizeof(err)));
        ds4_native_checkpoint incremental,full;ds4_ns_batch_checkpoint(snapshot.b,&incremental);
        assert(ds4_native_capture(&full,&target,err,sizeof(err)));
        assert(ds4_ns_checkpoint_same(&incremental,&full));
        /* Every serialized append slice must be byte-identical to the owner. */
        unsigned char actual[4096],expected[4096];
        for(uint32_t i=0;i<target.span_count;i++){
            uint64_t off=ds4_ns_append_slot(&target.spans[i])>=0?snapshot.b->base_bytes[i]:0;
            size_t n=(size_t)(target.spans[i].bytes-off);if(n>sizeof(actual))n=sizeof(actual);
            if(n){assert(ds4_ns_batch_span(snapshot.b,i,off,actual,n));assert(async_fixture_read(&target_fixture,i,off,expected,n));assert(!memcmp(actual,expected,n));}
        }
        ds4_ns_batch_free(snapshot.b);
    }
    base_fixture.corrupt=true;s.chain_valid=false;
    assert(!ds4_ns_chain_visit(&s,&base,err,sizeof(err))); /* changed prefix fails closed */
    base_fixture.corrupt=false;assert(ds4_ns_chain_visit(&s,&base,err,sizeof(err)));
    ds4_native_view target=async_fixture_view(&target_fixture,64,tokens,logits);
    ds4_ns_snap snapshot={&s,NULL,0};assert(ds4_ns_snapshot_visit(&snapshot,&target,err,sizeof(err)));
    snapshot.b->base=39; /* independent queued-base corruption: no socket is opened */
    uint32_t depth=0;assert(ds4_ns_q_push(&s,snapshot.b,&depth));
    assert(!ds4_ns_q_join(&s,err,sizeof(err)));
    assert(strstr(err,"catch-up base differs"));
    ds4_ns_q_stop(&s);assert(!s.q_init&&!s.q_thread_started&&!s.q_head&&!s.q_count);
    free(logits);
}
