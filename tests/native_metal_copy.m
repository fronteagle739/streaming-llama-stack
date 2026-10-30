/* Exercise actual dispatches and private encoder state, with no model loaded. */
#import "../engines/ds4-mac/ds4_metal.m"
#include <assert.h>
#include "../engines/ds4-mac/ds4_engram.h"

static const char *fixture_path(const char *name) {
    static char path[4096];
    const char *root=getenv("DS4_TEST_SCRATCH");
    assert(root && root[0]=='/');
    int n=snprintf(path,sizeof(path),"%s/%s",root,name);
    assert(n>0 && (size_t)n<sizeof(path));
    return path;
}

static void check_engram_format(void) {
    uint8_t raw[DS4_ENGRAM_ROW_BYTES];float ref[DS4_ENGRAM_DIM],got[DS4_ENGRAM_DIM];
    assert(!ds4_engram_decode_row(NULL,got) && errno==EINVAL);
    assert(!ds4_engram_decode_row(raw,NULL) && errno==EINVAL);
    for(unsigned scale=0;scale<256;scale++)for(unsigned code=0;code<256;code++) {
        memset(raw,code,DS4_ENGRAM_DIM);memset(raw+DS4_ENGRAM_DIM,scale,8);
        ds4_engram_set_format_lookup(0);errno=0;bool a=ds4_engram_decode_row(raw,ref);int ae=errno;
        ds4_engram_set_format_lookup(1);errno=0;bool b=ds4_engram_decode_row(raw,got);int be=errno;
        assert(a==b);
        if(a)assert(!memcmp(ref,got,sizeof(ref)));
        else assert(ae==EDOM && be==EDOM);
    }
    ds4_engram_set_format_lookup(1);
    puts("PASS: all 65536 E4M3/E8M0 code/scale pairs, all row words, signed zero, underflow, invalid/overflow EDOM, null EINVAL");
}

static void check_token_readers(void) {
    const char *path=fixture_path("native-engram-fixture.bin");
    unsigned char data[32*DS4_ENGRAM_ROW_BYTES];
    for(unsigned row=0;row<32;row++) {
        unsigned char *p=data+row*DS4_ENGRAM_ROW_BYTES;
        for(unsigned col=0;col<DS4_ENGRAM_DIM;col++) p[col]=(unsigned char)(((row+col)%112) | ((col&1)*128));
        memset(p+DS4_ENGRAM_DIM,127,8);
    }
    FILE *f=fopen(path,"w+b");assert(f);
    assert(fwrite(data,1,sizeof(data),f)==sizeof(data) && !fflush(f));
    ds4_engram_table tables[2];
    assert(ds4_engram_table_open(&tables[0],path,0,32));
    assert(ds4_engram_table_open(&tables[1],path,0,32));
    ds4_engram_reader *reader=ds4_engram_reader_create();assert(reader);
    float got[2][DS4_ENGRAM_COLS*DS4_ENGRAM_DIM], ref[2][DS4_ENGRAM_COLS*DS4_ENGRAM_DIM];
    uint32_t ids[2][DS4_ENGRAM_COLS];
    assert(!ds4_engram_reader_wait(reader,0) && errno==EINVAL);
    for(unsigned generation=0;generation<32;generation++) {
        for(unsigned t=0;t<2;t++) {
            for(unsigned col=0;col<DS4_ENGRAM_COLS;col++) ids[t][col]=(col*5+t*3+generation)%32;
            assert(ds4_engram_read(&tables[t],ids[t],DS4_ENGRAM_COLS,ref[t]));
        }
        assert(ds4_engram_reader_start(reader,tables,ids,(float *)got));
        assert(ds4_engram_reader_wait(reader,0));
        assert(!memcmp(got[0],ref[0],sizeof(ref[0])));
        assert(ds4_engram_reader_wait(reader,1));
        assert(!memcmp(got[1],ref[1],sizeof(ref[1])));
        uint64_t issue,done;
        assert(ds4_engram_reader_timing(reader,0,&issue,&done) && issue && done>=issue);
        assert(ds4_engram_reader_timing(reader,1,&issue,&done) && done>=issue);
    }
    ids[0][0]=32;
    assert(!ds4_engram_reader_start(reader,tables,ids,(float *)got) && errno==EINVAL);
    ids[0][0]=0;
    ds4_engram_table bad[2]={tables[0],tables[1]};bad[0].offset=sizeof(data)-100;
    assert(ds4_engram_reader_start(reader,bad,ids,(float *)got));
    assert(!ds4_engram_reader_wait(reader,0) && errno==EIO);
    assert(ds4_engram_reader_wait(reader,1));
    assert(!fseek(f,0,SEEK_SET) && fputc(127,f)!=EOF && !fflush(f));
    ids[0][0]=ids[1][0]=0;
    assert(ds4_engram_reader_start(reader,tables,ids,(float *)got));
    assert(!ds4_engram_reader_wait(reader,0) && errno==EDOM);
    assert(!ds4_engram_reader_wait(reader,1) && errno==EDOM);
    ds4_engram_reader_free(reader);
    assert(!fseek(f,0,SEEK_SET) && fwrite(data,1,sizeof(data),f)==sizeof(data) && !fflush(f));
    reader=ds4_engram_reader_create();assert(reader);
    assert(ds4_engram_reader_start(reader,tables,ids,(float *)got));
    ds4_engram_reader_free(reader); /* joins/cancels in-flight work before close */
    ds4_engram_table_close(&tables[0]);ds4_engram_table_close(&tables[1]);fclose(f);
    puts("PASS: persistent Engram 32 generations, exact rows, completion timestamps, EINVAL/EIO/EDOM and in-flight cleanup");
}

static void check_projections(void) {
    ds41_levers_init_from_env();
    const uint64_t bytes=8ull*1024u*4352u;
    unsigned char *weights=NULL;
    assert(!posix_memalign((void**)&weights,16384,bytes));
    for(uint64_t i=0;i<bytes;i+=34) {
        weights[i]=0; weights[i+1]=0x38; /* exact half scale 0.5 */
        for(int k=0;k<32;k++) weights[i+2+k]=(unsigned char)((int)((i/34+k*7)%31)-15);
    }
    FILE *f=fopen(fixture_path("native-metal-fixture.bin"),"w+b"); assert(f);
    assert(fwrite(weights,1,bytes,f)==bytes && fflush(f)==0);
    assert(ds4_gpu_set_model_fd_for_map(fileno(f),weights));
    assert(ds4_gpu_set_model_map(weights,bytes));
    ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(32768u*4), *a=ds4_gpu_tensor_alloc(8192u*4), *b=ds4_gpu_tensor_alloc(8192u*4);
    assert(x && a && b);
    float *xp=ds4_gpu_tensor_contents(x);
    for(int i=0;i<32768;i++) xp[i]=(float)(i%31-15)/16;
    for(int round=0;round<=1;round++) {
        g_ds41_levers.attn_low_static=0;
        assert(ds4_gpu_attention_output_low_q8_tensor(a,weights,bytes,0,4096,1024,8,x,round));
        g_ds41_levers.attn_low_static=1;
        assert(ds4_gpu_attention_output_low_q8_tensor(b,weights,bytes,0,4096,1024,8,x,round));
        assert(!memcmp(ds4_gpu_tensor_contents(a),ds4_gpu_tensor_contents(b),8192u*4));
    }
    assert(!ds4_gpu_dsv41_shared_down_overlap_begin()); /* no batch */
    assert(ds4_gpu_begin_commands());
    assert(ds4_gpu_dsv41_q8_virtual_down_tensor(a,weights,bytes,0,x));
    assert(ds4_gpu_end_commands());
    assert(ds4_gpu_begin_commands());
    assert(ds4_gpu_dsv41_shared_down_overlap_begin());
    assert(!ds4_gpu_dsv41_shared_down_overlap_begin()); /* nesting refused */
    assert(ds4_gpu_dsv41_q8_virtual_down_tensor(b,weights,bytes,0,x));
    assert(ds4_gpu_dsv41_q8_virtual_down_tensor(a,weights,bytes,0,x));
    assert(g_batch_enc && g_batch_enc.dispatchType==MTLDispatchTypeConcurrent);
    /* Independent work in the same concurrent encoder, then a serial consumer. */
    assert(ds4_gpu_tensor_copy(x,32752u*4,x,0,16)); /* retained concurrent blit */
    ds4_gpu_dsv41_shared_down_overlap_end();
    assert(!g_batch_encoder_concurrent && !g_dsv41_shared_down_overlap);
    assert(ds4_gpu_tensor_copy(b,5120u*4,a,0,1024));
    assert(ds4_gpu_end_commands());
    assert(!memcmp(ds4_gpu_tensor_contents(a),ds4_gpu_tensor_contents(b),5120u*4));
    assert(ds4_gpu_begin_commands() && ds4_gpu_dsv41_shared_down_overlap_begin());
    ds4_gpu_parallel_ffn_abort();
    assert(!g_batch_encoder_concurrent && !g_dsv41_shared_down_overlap);
    assert(ds4_gpu_end_commands());
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(a); ds4_gpu_tensor_free(b);
    /* Model views own their wired copies until GPU cleanup; original file stays. */
    free(weights); fclose(f);
    puts("PASS: generic/static low projection, both BF16 modes; shared-down overlap, serial consumer and abort");
}

static void check_verify_kernels(void) {
    enum { D=5120, FF=2304, NE=384, NU=6, ROWS=8, MXFP4=39 };
    const uint64_t gr=D/32*17, dr=FF/32*17, eb=FF*gr;
    const uint64_t eg=0, eu=NE*eb, ed=2*NE*eb, bytes=3*NE*eb;
    /* The API validates all 384 logical expert ranges. Keep the synthetic
     * model entirely in one owned wired RAM buffer: no multi-GB scratch file.
     * Only the selected six experts need nonzero payloads. This test runs
     * alone under a 12 GiB allocator budget and a 20 GiB outer RSS budget. */
    id<MTLBuffer> buffer=ds4_metal_owned_wired_buffer(g_device,bytes,ds4_gpu_model_resource_options(),NULL);
    assert(buffer && g_model_view_count < DS4_METAL_MAX_MODEL_VIEWS);
    unsigned char *map=buffer.contents;
    for(unsigned tensor=0;tensor<3;tensor++)
        for(uint64_t i=0;i<ROWS*NU*eb;i+=17) {
            unsigned char *p=map+tensor*NE*eb+i;
            p[0]=(unsigned char)(118+(i/17+tensor)%8);
            for(unsigned j=1;j<17;j++) p[j]=(unsigned char)((i/17)*137+j*23+tensor*41);
        }
    g_model_views[g_model_view_count++]=(ds4_gpu_model_view){.buffer=buffer,.model_map=map,.model_size=bytes,.model_offset=0,.bytes=bytes};
    ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(ROWS*D*4), *sel=ds4_gpu_tensor_alloc(ROWS*NU*4), *w=ds4_gpu_tensor_alloc(ROWS*NU*4);
    ds4_gpu_tensor *gate=ds4_gpu_tensor_alloc(ROWS*NU*FF*4), *up=ds4_gpu_tensor_alloc(ROWS*NU*FF*4), *mid=ds4_gpu_tensor_alloc(ROWS*NU*FF*4), *experts=ds4_gpu_tensor_alloc(ROWS*NU*D*4);
    ds4_gpu_tensor *ref=ds4_gpu_tensor_alloc(ROWS*D*4), *got=ds4_gpu_tensor_alloc(ROWS*D*4), *plain=ds4_gpu_tensor_alloc(ROWS*D*4);
    assert(x && sel && w && gate && up && mid && experts && ref && got && plain);
    float *xp=ds4_gpu_tensor_contents(x), *wp=ds4_gpu_tensor_contents(w);
    int32_t *sp=ds4_gpu_tensor_contents(sel);
    for(int i=0;i<ROWS*D;i++) xp[i]=(float)(i%31-15)/16;
    for(int pattern=0;pattern<2;pattern++) {
    for(int row=0;row<ROWS;row++) for(int k=0;k<NU;k++) { sp[row*NU+k]=pattern ? row*NU+k : (row+k)%NU; wp[row*NU+k]=(float)(k+1)/32; }
    g_ds41_levers.mxfp4_fixed_shapes=0;
    g_ds41_levers.mxfp4_dedup=0;
    for(unsigned row=0;row<ROWS;row++) {
        ds4_gpu_tensor *o=ds4_gpu_tensor_view(ref,row*D*4,D*4), *xr=ds4_gpu_tensor_view(x,row*D*4,D*4), *sr=ds4_gpu_tensor_view(sel,row*NU*4,NU*4), *wr=ds4_gpu_tensor_view(w,row*NU*4,NU*4);
        assert(o && xr && sr && wr);
        assert(ds4_gpu_routed_moe_one_tensor(o,gate,up,mid,experts,map,bytes,eg,eu,ed,MXFP4,MXFP4,eb,gr,eb,dr,D,FF,D,sr,wr,NE,NU,10.0f,xr,NULL,3,true));
        ds4_gpu_tensor_free(o); ds4_gpu_tensor_free(xr); ds4_gpu_tensor_free(sr); ds4_gpu_tensor_free(wr);
    }
    for(unsigned rows=2;rows<=ROWS;rows++) {
        bool half=false;
        g_ds41_levers.mxfp4_dedup=0;
        g_ds41_levers.verify_mxfp4=1; ds4_gpu_dsv41_verify_rows(1);
        assert(ds4_gpu_routed_moe_batch_tensor(got,gate,up,mid,experts,map,bytes,eg,eu,ed,MXFP4,MXFP4,eb,gr,eb,dr,D,FF,D,sel,w,NE,NU,10.0f,x,3,rows,&half,true));
        assert(!half && !memcmp(ds4_gpu_tensor_contents(ref),ds4_gpu_tensor_contents(got),rows*D*4));
        const size_t pair_bytes=rows*NU*FF*4;
        void *gate_ref=malloc(pair_bytes), *up_ref=malloc(pair_bytes), *mid_ref=malloc(pair_bytes);
        assert(gate_ref && up_ref && mid_ref);
        memcpy(gate_ref,ds4_gpu_tensor_contents(gate),pair_bytes);
        memcpy(up_ref,ds4_gpu_tensor_contents(up),pair_bytes);
        memcpy(mid_ref,ds4_gpu_tensor_contents(mid),pair_bytes);
        g_ds41_levers.mxfp4_dedup=1;
        assert(ds4_gpu_routed_moe_batch_tensor(got,gate,up,mid,experts,map,bytes,eg,eu,ed,MXFP4,MXFP4,eb,gr,eb,dr,D,FF,D,sel,w,NE,NU,10.0f,x,3,rows,&half,true));
        assert(g_dsv41_moe_dedup_plan && !half);
        assert(!memcmp(ds4_gpu_tensor_contents(ref),ds4_gpu_tensor_contents(got),rows*D*4));
        assert(!memcmp(gate_ref,ds4_gpu_tensor_contents(gate),pair_bytes));
        assert(!memcmp(up_ref,ds4_gpu_tensor_contents(up),pair_bytes));
        assert(!memcmp(mid_ref,ds4_gpu_tensor_contents(mid),pair_bytes));
        free(gate_ref); free(up_ref); free(mid_ref);
        /* The same rows outside DSpark retain the old general batch bits. */
        ds4_gpu_dsv41_verify_rows(0); g_ds41_levers.verify_mxfp4=0;
        assert(ds4_gpu_routed_moe_batch_tensor(plain,gate,up,mid,experts,map,bytes,eg,eu,ed,MXFP4,MXFP4,eb,gr,eb,dr,D,FF,D,sel,w,NE,NU,10.0f,x,3,rows,&half,true));
        g_ds41_levers.verify_mxfp4=1;
        assert(ds4_gpu_routed_moe_batch_tensor(got,gate,up,mid,experts,map,bytes,eg,eu,ed,MXFP4,MXFP4,eb,gr,eb,dr,D,FF,D,sel,w,NE,NU,10.0f,x,3,rows,&half,true));
        assert(!memcmp(ds4_gpu_tensor_contents(plain),ds4_gpu_tensor_contents(got),rows*D*4));
    }
    }
    const uint64_t qb=FF*(D/32*34), qbytes=3*qb;
    id<MTLBuffer> qbuffer=ds4_metal_owned_wired_buffer(g_device,qbytes,ds4_gpu_model_resource_options(),NULL);
    assert(qbuffer && g_model_view_count < DS4_METAL_MAX_MODEL_VIEWS);
    unsigned char *qm=qbuffer.contents;
    for(uint64_t i=0;i<qbytes;i+=34) {
        qm[i]=0; qm[i+1]=0x38;
        for(unsigned k=0;k<32;k++) qm[i+2+k]=(unsigned char)((int)((i/34+k*7)%31)-15);
    }
    g_model_views[g_model_view_count++]=(ds4_gpu_model_view){.buffer=qbuffer,.model_map=qm,.model_size=qbytes,.model_offset=0,.bytes=qbytes};
    ds4_gpu_tensor *sg=ds4_gpu_tensor_alloc(FF*4), *su=ds4_gpu_tensor_alloc(FF*4), *sm=ds4_gpu_tensor_alloc(FF*4), *so=ds4_gpu_tensor_alloc(D*4), *smref=ds4_gpu_tensor_alloc(FF*4), *soref=ds4_gpu_tensor_alloc(D*4);
    assert(sg && su && sm && so && smref && soref);
    assert(ds4_gpu_begin_commands());
    assert(ds4_gpu_dsv41_shared_mid_swiglu_q8_0_tensor(smref,qm,qbytes,0,qb,D,FF,x,10.0f,1.0f));
    assert(ds4_gpu_dsv41_q8_virtual_down_tensor(soref,qm,qbytes,2*qb,smref));
    assert(ds4_gpu_end_commands());
    for(int fixed=0;fixed<=1;fixed++) {
        g_ds41_levers.mxfp4_fixed_shapes=fixed;
        assert(ds4_gpu_begin_commands());
        assert(ds4_gpu_parallel_ffn_start_bf16(sg,su,sm,so,qm,qbytes,0,qb,2*qb,D,FF,x,10.0f));
        assert(ds4_gpu_routed_moe_one_tensor(got,gate,up,mid,experts,map,bytes,eg,eu,ed,MXFP4,MXFP4,eb,gr,eb,dr,D,FF,D,sel,w,NE,NU,10.0f,x,NULL,3,true));
        assert(ds4_gpu_parallel_ffn_finish());
        assert(ds4_gpu_end_commands());
        assert(!memcmp(ds4_gpu_tensor_contents(ref),ds4_gpu_tensor_contents(got),D*4));
        assert(!memcmp(ds4_gpu_tensor_contents(smref),ds4_gpu_tensor_contents(sm),FF*4));
        assert(!memcmp(ds4_gpu_tensor_contents(soref),ds4_gpu_tensor_contents(so),D*4));
    }
    assert(ds4_gpu_begin_commands());
    assert(ds4_gpu_parallel_ffn_start_bf16(sg,su,sm,so,qm,qbytes,0,qb,2*qb,D,FF,x,10.0f));
    assert(!ds4_gpu_routed_moe_one_tensor(NULL,gate,up,mid,experts,map,bytes,eg,eu,ed,MXFP4,MXFP4,eb,gr,eb,dr,D,FF,D,sel,w,NE,NU,10.0f,x,NULL,3,true));
    assert(!g_parallel_q8_pending && !g_batch_encoder_concurrent);
    assert(ds4_gpu_end_commands());
    ds4_gpu_tensor *split=ds4_gpu_tensor_alloc(24*4),*hr=ds4_gpu_tensor_alloc(4*D*4),*hg=ds4_gpu_tensor_alloc(4*D*4),
        *br=ds4_gpu_tensor_alloc(D*4),*bg=ds4_gpu_tensor_alloc(D*4),*pr=ds4_gpu_tensor_alloc(16),*pg=ds4_gpu_tensor_alloc(16);
    assert(split && hr && hg && br && bg && pr && pg);
    for(unsigned i=0;i<24;i++)((float *)ds4_gpu_tensor_contents(split))[i]=(float)((int)i-7)/32.f;
    assert(ds4_gpu_begin_commands());
    assert(ds4_gpu_dsv41_q8_virtual_down_tensor(soref,qm,qbytes,2*qb,smref));
    assert(ds4_gpu_dsv41_hc_round_expand4_tensor(hr,br,ref,soref,x,split,D,4));
    assert(ds4_gpu_tensor_copy(pr,0,split,0,16));
    assert(ds4_gpu_dsv41_shared_down_hc(hg,pg,bg,so,ref,x,split,smref,qm,qbytes,2*qb));
    assert(ds4_gpu_end_commands());
    assert(!memcmp(ds4_gpu_tensor_contents(soref),ds4_gpu_tensor_contents(so),D*4));
    assert(!memcmp(ds4_gpu_tensor_contents(br),ds4_gpu_tensor_contents(bg),D*4));
    assert(!memcmp(ds4_gpu_tensor_contents(hr),ds4_gpu_tensor_contents(hg),4*D*4));
    assert(!memcmp(ds4_gpu_tensor_contents(pr),ds4_gpu_tensor_contents(pg),16));
    assert(!ds4_gpu_dsv41_shared_down_hc(hg,pg,bg,NULL,ref,x,split,smref,qm,qbytes,2*qb));
    ds4_gpu_tensor *join[]={split,hr,hg,br,bg,pr,pg};
    for(unsigned i=0;i<sizeof(join)/sizeof(*join);i++)ds4_gpu_tensor_free(join[i]);
    puts("PASS: shared-down/HC join preserves all shared, block, HC and pre words; invalid buffers refuse");
    ds4_gpu_tensor *all[]={x,sel,w,gate,up,mid,experts,ref,got,plain,sg,su,sm,so,smref,soref};
    for(unsigned i=0;i<sizeof(all)/sizeof(*all);i++) ds4_gpu_tensor_free(all[i]);
    puts("PASS: MXFP4 verify rows 2-8 equal independent scalar rows; dedup gate/up/mid/output exact with shared and disjoint experts; ordinary batches unchanged");
    puts("PASS: full BF16 shared/routed overlap equals serial middle/down/routed words, both fixed-shape modes; early failure cleanup");
}

static void check_fixed_dense(void) {
    const uint64_t bytes=8192ull/32*34*5120;
    id<MTLBuffer> buffer=ds4_metal_owned_wired_buffer(g_device,bytes,ds4_gpu_model_resource_options(),NULL);
    assert(buffer && g_model_view_count<DS4_METAL_MAX_MODEL_VIEWS);
    unsigned char *map=buffer.contents;
    for(uint64_t i=0;i<bytes;i+=34) {
        map[i]=0;map[i+1]=0x30;
        for(unsigned j=2;j<34;j++)map[i+j]=(unsigned char)(i/34*17+j*11);
    }
    g_model_views[g_model_view_count++]=(ds4_gpu_model_view){.buffer=buffer,.model_map=map,.model_size=bytes,.model_offset=0,.bytes=bytes};
    ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(8192*4),*res=ds4_gpu_tensor_alloc(20480*4),*split=ds4_gpu_tensor_alloc(24*4);
    ds4_gpu_tensor *qa[2],*kv[2],*block[2],*out[2];
    for(unsigned i=0;i<2;i++) {
        qa[i]=ds4_gpu_tensor_alloc(1280*4);kv[i]=ds4_gpu_tensor_alloc(512*4);
        block[i]=ds4_gpu_tensor_alloc(5120*4);out[i]=ds4_gpu_tensor_alloc(20480*4);
        assert(qa[i] && kv[i] && block[i] && out[i]);
    }
    assert(x && res && split);
    for(unsigned i=0;i<8192;i++)((float *)ds4_gpu_tensor_contents(x))[i]=(float)((int)(i*11%127)-63)/64.f;
    for(unsigned i=0;i<20480;i++)((float *)ds4_gpu_tensor_contents(res))[i]=(float)((int)(i*19%127)-63)/64.f;
    for(unsigned i=0;i<24;i++)((float *)ds4_gpu_tensor_contents(split))[i]=(float)(i+1)/32.f;
    for(unsigned i=0;i<2;i++) {
        g_ds41_levers.fixed_dense=i;
        assert(ds4_gpu_begin_commands());
        assert(ds4_gpu_dsv41_arch_qa_kv_tensor(qa[i],kv[i],x,map,bytes,0,5120ull/32*34*1280));
        assert(ds4_gpu_dsv41_attn_out_hc_tensor(out[i],block[i],map,bytes,0,x,res,split));
        assert(ds4_gpu_end_commands());
    }
    assert(!memcmp(ds4_gpu_tensor_contents(qa[0]),ds4_gpu_tensor_contents(qa[1]),1280*4));
    assert(!memcmp(ds4_gpu_tensor_contents(kv[0]),ds4_gpu_tensor_contents(kv[1]),512*4));
    assert(!memcmp(ds4_gpu_tensor_contents(block[0]),ds4_gpu_tensor_contents(block[1]),5120*4));
    assert(!memcmp(ds4_gpu_tensor_contents(out[0]),ds4_gpu_tensor_contents(out[1]),20480*4));
    g_ds41_levers.fixed_dense=0;
    for(unsigned i=0;i<2;i++) { ds4_gpu_tensor_free(qa[i]);ds4_gpu_tensor_free(kv[i]);ds4_gpu_tensor_free(block[i]);ds4_gpu_tensor_free(out[i]); }
    ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(res);ds4_gpu_tensor_free(split);
    puts("PASS: literal/general QA, KV, attention block and all HC stream words are bit-identical");
}

static void check_heads_rope(void) {
    enum { HEADS=64,DIM=512,NWG=32,WORDS=HEADS*DIM*NWG+HEADS*NWG*2 };
    ds4_gpu_tensor *partial=ds4_gpu_tensor_alloc(WORDS*4),*plain=ds4_gpu_tensor_alloc(HEADS*DIM*4),
        *got=ds4_gpu_tensor_alloc(HEADS*DIM*4),*tab=ds4_gpu_tensor_alloc(512);
    ds4_gpu_tensor *qref=ds4_gpu_tensor_alloc(128*32*4),*qgot=ds4_gpu_tensor_alloc(128*32*4);
    float queries[128*32];
    for(unsigned i=0;i<128*32;i++)queries[i]=(float)((int)(i*137%251)-125)/17.f;
    assert(partial && plain && got && tab);
    assert(qref && qgot);
    float *values=ds4_gpu_tensor_contents(partial);
    for(unsigned i=0;i<HEADS*DIM*NWG;i++)values[i]=(float)((int)(i*137u%251)-125)/64.f;
    for(unsigned h=0;h<HEADS;h++)for(unsigned w=0;w<NWG;w++) {
        values[HEADS*DIM*NWG+h*2*NWG+2*w]=(float)(w+1)/32.f;
        values[HEADS*DIM*NWG+h*2*NWG+2*w+1]=(float)((int)(h+w)%17-8)/4.f;
    }
    const uint32_t positions[]={0,8192,131072,1048575};
    for(unsigned k=0;k<4;k++)for(unsigned comp=0;comp<2;comp++) {
        assert(ds4_gpu_begin_commands());
        id<MTLComputeCommandEncoder> enc=ds4_gpu_compute_encoder(g_batch_cb);
        id<MTLComputePipelineState> p=ds4_gpu_get_flash_attn_reduce_pipeline(DIM,NWG);assert(p);
        ds4_gpu_flash_attn_reduce_args a={.nrows=HEADS};
        [enc setComputePipelineState:p];[enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:ds4_gpu_tensor_buffer(partial) offset:ds4_gpu_tensor_offset(partial) atIndex:1];
        [enc setBuffer:ds4_gpu_tensor_buffer(plain) offset:ds4_gpu_tensor_offset(plain) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(HEADS,1,1) threadsPerThreadgroup:MTLSizeMake(32*NWG,1,1)];
        ds4_gpu_end_compute_encoder(g_batch_cb,enc);
        assert(ds4_gpu_dsv41_quantize(plain,DIM,HEADS,DS4_V41_BF16));
        assert(ds4_gpu_dsv41_rope(plain,DIM,HEADS,1,positions[k],comp,true));
        assert(ds4_gpu_dsv41_step_rope_table(tab,positions[k]));
        ds4_gpu_dsv41_heads_once_arm(tab,comp);
        int transformed=0;p=ds4_gpu_heads_once_reduce(p,HEADS,DIM,NWG,&transformed);
        assert(p && transformed);
        enc=ds4_gpu_compute_encoder(g_batch_cb);
        [enc setComputePipelineState:p];[enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:ds4_gpu_tensor_buffer(partial) offset:ds4_gpu_tensor_offset(partial) atIndex:1];
        [enc setBuffer:ds4_gpu_tensor_buffer(got) offset:ds4_gpu_tensor_offset(got) atIndex:2];
        ds4_gpu_heads_once_bind(enc);
        [enc dispatchThreadgroups:MTLSizeMake(HEADS,1,1) threadsPerThreadgroup:MTLSizeMake(32*NWG,1,1)];
        ds4_gpu_end_compute_encoder(g_batch_cb,enc);
        assert(ds4_gpu_dsv41_heads_once_take() && ds4_gpu_end_commands());
        assert(!memcmp(ds4_gpu_tensor_contents(plain),ds4_gpu_tensor_contents(got),HEADS*DIM*4));
        for(unsigned format=1;format<=3;format++) {
            assert(ds4_gpu_tensor_write(qref,0,queries,sizeof(queries)));
            assert(ds4_gpu_tensor_write(qgot,0,queries,sizeof(queries)));
            assert(ds4_gpu_begin_commands());
            assert(ds4_gpu_dsv41_rope(qref,128,32,1,positions[k],comp,false));
            assert(ds4_gpu_dsv41_quantize(qref,128,32,(ds4_v41_activation_format)format));
            assert(ds4_gpu_dsv41_rope_quantize_heads_tab(qgot,128,32,positions[k],comp,(ds4_v41_activation_format)format,tab));
            assert(ds4_gpu_end_commands());
            assert(!memcmp(ds4_gpu_tensor_contents(qref),ds4_gpu_tensor_contents(qgot),sizeof(queries)));
        }
    }
    ds4_gpu_tensor_free(partial);ds4_gpu_tensor_free(plain);ds4_gpu_tensor_free(got);ds4_gpu_tensor_free(tab);
    ds4_gpu_tensor_free(qref);ds4_gpu_tensor_free(qgot);
    puts("PASS: split reduction with BF16/inverse RoPE equals standalone kernels at positions 0/8K/131K/1048575, raw and compressed frequencies");
    puts("PASS: table index-query RoPE/quantization equals separate kernels at all four positions, both frequencies and all three quantization formats");
}

static void check_preencode(void) {
    enum { D=64, HC=4 };
    id<MTLBuffer> buffer=ds4_metal_owned_wired_buffer(g_device,16384,ds4_gpu_model_resource_options(),NULL);
    assert(buffer && g_model_view_count<DS4_METAL_MAX_MODEL_VIEWS);
    uint16_t *weights=buffer.contents;
    for(unsigned i=0;i<D;i++) { weights[i]=0x3c00;weights[D+i]=0x4000; }
    g_model_views[g_model_view_count++]=(ds4_gpu_model_view){.buffer=buffer,.model_map=weights,.model_size=16384,.model_offset=0,.bytes=16384};
    ds4_gpu_tensor *out=ds4_gpu_tensor_alloc(D*HC*4);assert(out);
    g_ds41_levers.preencode=1;
    for(unsigned mode=0;mode<7;mode++) {
        assert(ds4_gpu_begin_commands());
        assert(ds4_gpu_embed_token_hc_tensor(out,weights,16384,0,2,0,D,HC));
        assert(ds4_gpu_end_commands_commit(mode==0));
        assert(ds4_gpu_pe_begin());
        ds4_gpu_pe_embed_word(1);
        assert(ds4_gpu_embed_token_hc_tensor(out,weights,16384,0,2,0,D,HC));
        ds4_gpu_pe_embed_word(0);
        if(mode==3) { float x=0;assert(!ds4_gpu_tensor_write(out,0,&x,4)); }
        if(mode==4) assert(!ds4_gpu_flush_commands());
        if(mode==5) { float x;assert(!ds4_gpu_tensor_read(out,0,&x,4)); }
        if(mode==6) assert(!ds4_gpu_tensor_contents(out));
        assert(ds4_gpu_pe_end(1)==(mode<3));
        assert(ds4_gpu_end_commands_wait());
        for(unsigned i=0;i<D*HC;i++) assert(((float *)ds4_gpu_tensor_contents(out))[i]==1.0f);
        if(mode==0) {
            assert(ds4_gpu_pe_pending() && ds4_gpu_pe_commit(1));
            assert(ds4_gpu_end_commands());
            for(unsigned i=0;i<D*HC;i++) assert(((float *)ds4_gpu_tensor_contents(out))[i]==2.0f);
        } else if(mode==1) {
            ds4_gpu_pe_drop();assert(!ds4_gpu_pe_pending());
        } else if(mode==2) {
            assert(ds4_gpu_begin_commands());
            assert(!ds4_gpu_pe_pending()); /* foreign commands invalidate held work */
            assert(ds4_gpu_end_commands());
        } else assert(!ds4_gpu_pe_pending());
    }
    g_ds41_levers.preencode=0;
    assert(!ds4_gpu_pe_supported());
    ds4_gpu_tensor_free(out);
    puts("PASS: held preencode never executes early; actual embedding reads committed token word; explicit/foreign drops; host-write/flush failure cleanup");
}

static void check_copy(ds4_gpu_tensor *dst, uint64_t d,
                       ds4_gpu_tensor *src, uint64_t s, uint64_t n,
                       int compute, int concurrent) {
    unsigned char expected[8192], actual[8192];
    assert(n <= sizeof(expected));
    assert(ds4_gpu_tensor_read(src,s,expected,n));
    assert(ds4_gpu_begin_commands());
    if(concurrent) g_batch_encoder_concurrent=YES;
    assert(ds4_gpu_tensor_copy(dst,d,src,s,n));
    /* A compute copy leaves its serial encoder open; blit closes it. */
    assert((g_batch_enc != nil)==compute);
    if(concurrent) g_batch_encoder_concurrent=NO;
    assert(ds4_gpu_end_commands());
    assert(ds4_gpu_tensor_read(dst,d,actual,n));
    assert(memcmp(actual,expected,n)==0);
}


static void check_engram_gate(void) {
    const unsigned W=5120;
    for(unsigned rows=1;rows<=16;rows=rows==1?6:rows==6?16:17) {
        const size_t n=(size_t)W*rows*4, k=(size_t)W*rows*5;
        float *h=malloc(n*4),*kv=malloc(k*4),*w=malloc(W*4*4),*ref=malloc(n*4),*got=malloc(n*4);
        assert(h&&kv&&w&&ref&&got);
        for(size_t i=0;i<n;i++)h[i]=(float)((int)(i*37%1001)-500)/512.f;
        for(size_t i=0;i<k;i++)kv[i]=(float)((int)(i*19%733)-366)/512.f;
        for(size_t i=0;i<W*4;i++)w[i]=(float)((int)(i*13%251)-125)/256.f;
        ds4_gpu_tensor *a=ds4_gpu_tensor_alloc(n*4),*b=ds4_gpu_tensor_alloc(k*4),*c=ds4_gpu_tensor_alloc(W*4*4);
        assert(a&&b&&c);
        assert(ds4_gpu_tensor_write(b,0,kv,k*4)&&ds4_gpu_tensor_write(c,0,w,W*4*4));
        for(unsigned arm=0;arm<2;arm++) {
            g_ds41_levers.engram_gate=arm;
            assert(ds4_gpu_tensor_write(a,0,h,n*4));
            assert(ds4_gpu_dsv41_engram_add(a,b,c,c,NULL,W,rows,1.e-6f));
            assert(ds4_gpu_tensor_read(a,0,arm?got:ref,n*4));
        }
        assert(!memcmp(ref,got,n*4));
        ds4_gpu_tensor_free(a);ds4_gpu_tensor_free(b);ds4_gpu_tensor_free(c);
        free(h);free(kv);free(w);free(ref);free(got);
    }
    g_ds41_levers.engram_gate=0;
    puts("PASS: segmented Engram gate every word at 1/6/16 rows");
}


static void check_row_matmuls(void) {
    const unsigned widths[]={1280,5120,8192};
    ds41_levers_init_from_env();
    int old_pair=g_ds41_levers.mtp_q8_pair6,old_stream=g_ds41_levers.mtp_q8_stream6;
    g_ds41_levers.mtp_q8_pair6=0;g_ds41_levers.mtp_q8_stream6=0;
    for(unsigned shape=0;shape<3;shape++) {
        enum { N=40 };const unsigned K=widths[shape];size_t bytes=(size_t)N*(K/32)*34;
        id<MTLBuffer> owned=ds4_metal_owned_wired_buffer(g_device,bytes,ds4_gpu_model_resource_options(),NULL);
        assert(owned && g_model_view_count<DS4_METAL_MAX_MODEL_VIEWS);
        unsigned char *map=owned.contents;
        for(size_t i=0;i<bytes;i+=34) {
            map[i]=0;map[i+1]=0x38;
            for(unsigned j=0;j<32;j++)map[i+2+j]=(unsigned char)((int)((i/34+j*7)%31)-15);
        }
        g_model_views[g_model_view_count++]=(ds4_gpu_model_view){.buffer=owned,.model_map=map,.model_size=bytes,.model_offset=0,.bytes=bytes};
        ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(K*8*4),*out=ds4_gpu_tensor_alloc(N*8*4);
        assert(x&&out);float *in=malloc(K*8*4),ref[N*8],got[N*8];assert(in);
        for(unsigned i=0;i<K*8;i++)in[i]=(float)((int)(i*17%997)-498)/512.f;
        assert(ds4_gpu_tensor_write(x,0,in,K*8*4));
        for(unsigned rows=2;rows<=8;rows++)for(int round=0;round<2;round++) {
            for(int arm=0;arm<2;arm++) {
                g_ds41_levers.row_matmuls=arm;
                assert(ds4_gpu_matmul_q8_0_decode_rows_exact_impl(out,map,bytes,0,K,N,x,rows,round));
                assert(ds4_gpu_tensor_read(out,0,arm?got:ref,rows*N*4));
            }
            assert(!memcmp(ref,got,rows*N*4));
            g_ds41_levers.rows_mma=1;
            assert(!g_dsv41_dspark_rows);
            uint64_t before=g_dsv41_rows_mma_dispatches;
            assert(ds4_gpu_matmul_q8_0_decode_rows_exact_impl(out,map,bytes,0,K,N,x,rows,round));
            assert(ds4_gpu_tensor_read(out,0,got,rows*N*4) && !memcmp(ref,got,rows*N*4));
            assert(before==g_dsv41_rows_mma_dispatches);
            assert(ds4_gpu_dsv41_dspark_rows(1)==0);
            assert(ds4_gpu_matmul_q8_0_decode_rows_exact_impl(out,map,bytes,0,K,N,x,rows,round));
            assert(ds4_gpu_tensor_read(out,0,got,rows*N*4));
            assert(ds4_gpu_dsv41_dspark_rows(0)==1);
            bool eligible=rows>=(K>=8192?4u:3u);
            assert(g_dsv41_rows_mma_dispatches==before+eligible);
            float scale=0;for(unsigned i=0;i<rows*N;i++)scale=fmaxf(scale,fabsf(ref[i]));
            for(unsigned i=0;i<rows*N;i++)assert(isfinite(got[i]) && fabsf(got[i]-ref[i])<=1.e-5f*scale+(round?fabsf(ref[i])/128.f:0.f));
            if(rows==6 && !round && shape==0) {
                before=g_dsv41_rows_mma_dispatches;
                g_ds41_levers.mtp_q8_pair6=1;
                assert(ds4_gpu_dsv41_dspark_rows(1)==0);
                assert(!ds4_gpu_matmul_q8_0_decode_rows_exact_impl(out,map,bytes,0,K,N,x,rows,round));
                assert(ds4_gpu_dsv41_dspark_rows(0)==1 && before==g_dsv41_rows_mma_dispatches);
                g_ds41_levers.mtp_q8_pair6=0;
            }
            g_ds41_levers.rows_mma=0;
        }
        free(in);ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(out);
    }
    g_ds41_levers.row_matmuls=0;
    g_ds41_levers.mtp_q8_pair6=old_pair;g_ds41_levers.mtp_q8_stream6=old_stream;
    puts("PASS: exact Q8 row reuse; DSpark MMA eligible witnesses and upstream float/BF16 bounds; ordinary rows remain bit-identical");
}

int main(void) {
    @autoreleasepool {
        check_engram_format();
        check_token_readers();
        unsetenv("DS4_METAL_MODEL_UNTRACKED");
        unsetenv("DS4_METAL_MODEL_TRACKED");
        ds4_gpu_set_model_untracked_default(0);
        assert(!(ds4_gpu_model_resource_options() & MTLResourceHazardTrackingModeUntracked));
        ds4_gpu_set_model_untracked_default(1);
        assert(ds4_gpu_model_resource_options() & MTLResourceHazardTrackingModeUntracked);
        setenv("DS4_METAL_MODEL_TRACKED","1",1);
        assert(!(ds4_gpu_model_resource_options() & MTLResourceHazardTrackingModeUntracked));
        setenv("DS4_METAL_MODEL_TRACKED","0",1);
        assert(ds4_gpu_model_resource_options() & MTLResourceHazardTrackingModeUntracked);
        setenv("DS4_METAL_MODEL_TRACKED","1",1);
        setenv("DS4_METAL_MODEL_UNTRACKED","0",1);
        assert(ds4_gpu_model_resource_options() & MTLResourceHazardTrackingModeUntracked);
        unsetenv("DS4_METAL_MODEL_TRACKED"); unsetenv("DS4_METAL_MODEL_UNTRACKED");
        assert(ds4_gpu_init());
        ds4_gpu_tensor *a=ds4_gpu_tensor_alloc(16384), *b=ds4_gpu_tensor_alloc(16384);
        assert(a && b);
        unsigned char pattern[16384];
        for(size_t i=0;i<sizeof(pattern);i++) pattern[i]=(i*137+(i>>3)*23)&255;
        assert(ds4_gpu_tensor_write(a,0,pattern,sizeof(pattern)));
        unsetenv("DS4_METAL_DISABLE_COMPUTE_TENSOR_COPY");
        check_copy(b,0,a,0,16,1,0);
        check_copy(b,32,a,16,8176,1,0);
        ds4_gpu_tensor *av=ds4_gpu_tensor_view(a,32,8192);
        assert(av); check_copy(b,16,av,16,4096,1,0);
        check_copy(a,8192,a,0,4096,1,0);
        check_copy(b,3,a,1,255,0,0);
        check_copy(b,0,a,0,4096,0,1);
        setenv("DS4_METAL_DISABLE_COMPUTE_TENSOR_COPY","1",1);
        check_copy(b,0,a,0,4096,0,0);
        setenv("DS4_METAL_DISABLE_COMPUTE_TENSOR_COPY","0",1);
        check_copy(b,0,a,0,4096,1,0);
        /* Two dependent copies in the same serial encoder exercise ordering. */
        assert(ds4_gpu_begin_commands());
        assert(ds4_gpu_tensor_copy(b,0,a,0,4096));
        assert(ds4_gpu_tensor_copy(b,8192,b,0,4096));
        assert(g_batch_enc && ds4_gpu_end_commands());
        assert(!memcmp(ds4_gpu_tensor_contents(b),(char*)ds4_gpu_tensor_contents(b)+8192,4096));
        assert(ds4_gpu_tensor_copy(b,16384,a,16384,0));
        assert(!ds4_gpu_tensor_copy(b,16385,a,0,0));
        assert(!ds4_gpu_tensor_copy(b,16384,a,0,1));
        assert(!ds4_gpu_tensor_copy(b,0,a,16384,1));
        assert(!ds4_gpu_tensor_copy(b,0,a,0,16)); /* no command batch */
        ds4_gpu_tensor_free(av); ds4_gpu_tensor_free(a); ds4_gpu_tensor_free(b);
        check_row_matmuls();
        check_engram_gate();
        check_preencode();
        check_heads_rope();
        check_fixed_dense();
        check_projections();
        check_verify_kernels();
        ds4_gpu_cleanup();
        puts("PASS: model-view policy; compute/blit/concurrent copies; exact bytes; ordering; bounds");
    }
    return 0;
}
