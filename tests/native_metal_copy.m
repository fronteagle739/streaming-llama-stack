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
    const uint64_t hbytes=20480ull*24*2+16+96,hs=20480ull*24*2,hb=hs+16;
    id<MTLBuffer> hbuffer=ds4_metal_owned_wired_buffer(g_device,hbytes,ds4_gpu_model_resource_options(),NULL);
    assert(hbuffer&&g_model_view_count<DS4_METAL_MAX_MODEL_VIEWS);unsigned char *hm=hbuffer.contents;
    uint16_t *hw=(uint16_t *)hm;for(unsigned i=0;i<hs/2;i++)hw[i]=(i&1)?0x1c00:0x9c00;
    float *hv=(float *)(hm+hs);for(unsigned i=0;i<4;i++)hv[i]=.05f;
    hv=(float *)(hm+hb);for(unsigned i=0;i<24;i++)hv[i]=(float)((int)(i%7)-3)/32.f;
    g_model_views[g_model_view_count++]=(ds4_gpu_model_view){.buffer=hbuffer,.model_map=hm,.model_size=hbytes,.model_offset=0,.bytes=hbytes};
    ds4_gpu_tensor *hmix[2],*hsplit[2],*hcounter=ds4_gpu_tensor_alloc(4);assert(hcounter);
    for(unsigned i=0;i<2;i++) {hmix[i]=ds4_gpu_tensor_alloc(96);hsplit[i]=ds4_gpu_tensor_alloc(96);assert(hmix[i]&&hsplit[i]);}
    uint32_t hz=0;assert(ds4_gpu_tensor_write(hcounter,0,&hz,4)&&ds4_gpu_begin_commands());
    assert(ds4_gpu_dsv41_mtp_hc_tensor(hmix[0],hsplit[0],hcounter,x,hm,hbytes,0,hs,hb,20,1.e-6f,1.e-6f));
    assert(ds4_gpu_end_commands());
    for(int fixed=0;fixed<=1;fixed++)for(int hc=0;hc<=1;hc++) {
        g_ds41_levers.mxfp4_fixed_shapes=fixed;
        assert(ds4_gpu_tensor_write(hcounter,0,&hz,4));
        assert(ds4_gpu_begin_commands());
        assert(ds4_gpu_parallel_ffn_start_bf16(sg,su,sm,so,qm,qbytes,0,qb,2*qb,D,FF,x,10.0f));
        if(hc) {
            assert(ds4_gpu_dsv41_mtp_hc_tensor(hmix[1],hsplit[1],hcounter,x,hm,hbytes,0,hs,hb,20,1.e-6f,1.e-6f));
            assert(g_batch_encoder_concurrent && g_batch_enc.dispatchType==MTLDispatchTypeConcurrent);
        }
        assert(ds4_gpu_routed_moe_one_tensor(got,gate,up,mid,experts,map,bytes,eg,eu,ed,MXFP4,MXFP4,eb,gr,eb,dr,D,FF,D,sel,w,NE,NU,10.0f,x,NULL,3,true));
        assert(ds4_gpu_parallel_ffn_finish());
        assert(ds4_gpu_end_commands());
        assert(!memcmp(ds4_gpu_tensor_contents(ref),ds4_gpu_tensor_contents(got),D*4));
        assert(!memcmp(ds4_gpu_tensor_contents(smref),ds4_gpu_tensor_contents(sm),FF*4));
        assert(!memcmp(ds4_gpu_tensor_contents(soref),ds4_gpu_tensor_contents(so),D*4));
        if(hc) { assert(!memcmp(ds4_gpu_tensor_contents(hmix[0]),ds4_gpu_tensor_contents(hmix[1]),96));
            assert(!memcmp(ds4_gpu_tensor_contents(hsplit[0]),ds4_gpu_tensor_contents(hsplit[1]),96)); }
    }
    for(unsigned i=0;i<2;i++) {ds4_gpu_tensor_free(hmix[i]);ds4_gpu_tensor_free(hsplit[i]);}ds4_gpu_tensor_free(hcounter);
    assert(ds41_levers_set("concurrent_ffn",2));assert(!ds41_levers_set("concurrent_ffn",3));assert(ds41_levers_set("concurrent_ffn",0));
    puts("PASS: deferred HC mix/split concurrent with complete BF16 shared/routed FFN, every word exact and actual concurrent dispatch witnessed in both fixed-shape modes");
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


static void check_verify_indexed(void) {
    enum { K=1037, ROWS=17, HEADS=32, D=128 };
    const size_t kb=K*D*4,qb=ROWS*HEADS*D*4,wb=ROWS*HEADS*4,sb=ROWS*K*4;
    ds4_gpu_tensor *keys=ds4_gpu_tensor_alloc(kb),*view=ds4_gpu_tensor_alloc(kb/2),*flag=ds4_gpu_tensor_alloc(4),
        *q=ds4_gpu_tensor_alloc(qb),*weights=ds4_gpu_tensor_alloc(wb),*ref=ds4_gpu_tensor_alloc(sb),*got=ds4_gpu_tensor_alloc(sb),
        *mask=ds4_gpu_tensor_alloc(ROWS*((K+7)/8)*4);
    assert(keys&&view&&flag&&q&&weights&&ref&&got&&mask);
    uint32_t *kv=malloc(kb);float *qv=malloc(qb),*wv=malloc(wb),*rv=malloc(sb),*gv=malloc(sb);
    assert(kv&&qv&&wv&&rv&&gv);
    for(unsigned i=0;i<K*D;i++) { float f=(float)((int)(i*17%997)-498)/512.f;memcpy(kv+i,&f,4);kv[i]&=0xffff0000u; }
    kv[0]=0x80000000u;
    for(unsigned i=0;i<ROWS*HEADS*D;i++)qv[i]=(float)((int)(i*31%733)-366)/512.f;
    for(unsigned i=0;i<ROWS*HEADS;i++)wv[i]=(float)(i%23+1)/32.f;
    uint32_t zero=0,check=7;
    assert(ds4_gpu_tensor_write(keys,0,kv,kb)&&ds4_gpu_tensor_write(q,0,qv,qb)&&ds4_gpu_tensor_write(weights,0,wv,wb));
    assert(ds4_gpu_tensor_write(flag,0,&zero,4));
    assert(ds4_gpu_dsv41_index_keys_bf16(view,keys,0,K,flag));
    assert(ds4_gpu_tensor_read(flag,0,&check,4)&&check==0);
    const uint16_t *packed=ds4_gpu_tensor_contents(view);
    for(unsigned i=0;i<K*D;i++)assert(packed[i]==(kv[i]>>16));
    kv[127]|=1u;assert(ds4_gpu_tensor_write(keys,0,kv,kb));
    assert(ds4_gpu_dsv41_index_keys_bf16(view,keys,0,1,flag));
    assert(ds4_gpu_tensor_read(flag,0,&check,4)&&check==1);
    kv[127]&=0xffff0000u;assert(ds4_gpu_tensor_write(keys,0,kv,kb));
    assert(ds4_gpu_tensor_write(flag,0,&zero,4)&&ds4_gpu_dsv41_index_keys_bf16(view,keys,0,K,flag));
    assert(!ds4_gpu_dsv41_index_keys_bf16(view,keys,K,1,flag));
    int old=g_ds41_levers.verify_batch_core;g_ds41_levers.verify_batch_core=2;
    assert(ds4_gpu_dsv41_dspark_rows(1)==0);
    for(unsigned rows=2;rows<=ROWS;rows++) {
        assert(ds4_gpu_dsv41_indexer_scores_batch(ref,q,weights,keys,K,rows,2048,2));
        assert(ds4_gpu_dsv41_indexer_scores_rows_bf16(got,q,weights,view,NULL,0,rows,K,rows,2048,2));
        assert(ds4_gpu_tensor_read(ref,0,rv,rows*K*4)&&ds4_gpu_tensor_read(got,0,gv,rows*K*4));
        assert(!memcmp(rv,gv,rows*K*4));
    }
    // Every row drops block0; the shared block read must then be skipped.
    float *mv=ds4_gpu_tensor_contents(mask);memset(mv,0,ROWS*((K+7)/8)*4);
    for(unsigned i=0;i<ROWS;i++)mv[i*((K+7)/8)]=1.f;
    assert(ds4_gpu_dsv41_indexer_scores_rows_bf16(got,q,weights,view,mask,(K+7)/8,0,K,ROWS,2048,2));
    assert(ds4_gpu_tensor_read(got,0,gv,sb));
    for(unsigned r=0;r<ROWS;r++)for(unsigned k=0;k<K;k++) {
        if(k<8)assert(isinf(gv[r*K+k])&&gv[r*K+k]<0);
        else assert(!memcmp(gv+r*K+k,rv+r*K+k,4));
    }
    assert(ds4_gpu_dsv41_dspark_rows(0)==1);
    assert(!ds4_gpu_dsv41_indexer_scores_rows_bf16(got,q,weights,view,NULL,0,ROWS,K,ROWS,2048,2));
    assert(!ds4_gpu_dsv41_indexer_scores_batch(ref,q,weights,keys,K,ROWS,K*2,2));
    g_ds41_levers.verify_batch_core=old;
    ds4_gpu_tensor *all[]={keys,view,flag,q,weights,ref,got,mask};for(unsigned i=0;i<8;i++)ds4_gpu_tensor_free(all[i]);
    free(kv);free(qv);free(wv);free(rv);free(gv);
    puts("PASS: exact derived BF16 keys including signed zero and nonexact restore flag; 2-17-row scores match tiled F32; mask, scope and bounds");
}


static void check_verify_attention_rows(void) {
    enum { H=64,D=512,ROWS=8,K=1037,RAW=127+ROWS,TOP=512 };
    id<MTLBuffer> owned=ds4_metal_owned_wired_buffer(g_device,H*4,ds4_gpu_model_resource_options(),NULL);
    assert(owned && g_model_view_count<DS4_METAL_MAX_MODEL_VIEWS);
    float *sinks=owned.contents;for(unsigned h=0;h<H;h++)sinks[h]=(float)((int)h-31)/64.f;
    g_model_views[g_model_view_count++]=(ds4_gpu_model_view){.buffer=owned,.model_map=sinks,.model_size=H*4,.model_offset=0,.bytes=H*4};
    ds4_gpu_tensor *q=ds4_gpu_tensor_alloc(ROWS*H*D*4),*raw=ds4_gpu_tensor_alloc(RAW*D*4),*comp=ds4_gpu_tensor_alloc(K*D*4),
        *ids=ds4_gpu_tensor_alloc(ROWS*TOP*4),*out=ds4_gpu_tensor_alloc(ROWS*H*D*4),*ref=ds4_gpu_tensor_alloc(H*D*4);
    assert(q&&raw&&comp&&ids&&out&&ref);
    float *qv=ds4_gpu_tensor_contents(q),*rv=ds4_gpu_tensor_contents(raw),*cv=ds4_gpu_tensor_contents(comp);
    int32_t *iv=ds4_gpu_tensor_contents(ids);
    for(unsigned i=0;i<ROWS*H*D;i++)qv[i]=(float)((int)(i*17%733)-366)/512.f;
    for(unsigned i=0;i<RAW*D;i++)rv[i]=(float)((int)(i*19%997)-498)/512.f;
    for(unsigned i=0;i<K*D;i++)cv[i]=(float)((int)(i*37%991)-495)/512.f;
    for(unsigned r=0;r<ROWS;r++)for(unsigned j=0;j<TOP;j++)iv[r*TOP+j]=(j*17+r*7)%1024;
    ds41_levers_init_from_env();int old=g_ds41_levers.verify_batch_core;g_ds41_levers.verify_batch_core=2;
    assert(ds4_gpu_dsv41_dspark_rows(1)==0);
    assert(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(out,sinks,H*4,0,q,raw,comp,0,ids,ROWS,2048,RAW,RAW,0,K,TOP,128,2,H,D));
    const float *all=ds4_gpu_tensor_contents(out);
    for(unsigned r=0;r<ROWS;r++) {
        ds4_gpu_tensor *qr=ds4_gpu_tensor_view(q,(uint64_t)r*H*D*4,H*D*4),
            *rr=ds4_gpu_tensor_view(raw,(uint64_t)r*D*4,128*D*4),
            *ir=ds4_gpu_tensor_view(ids,(uint64_t)r*TOP*4,TOP*4);
        assert(qr&&rr&&ir);
        assert(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(ref,sinks,H*4,0,qr,rr,comp,0,ir,1,2048+r,128,128,0,K,TOP,128,2,H,D));
        assert(!memcmp(all+(size_t)r*H*D,ds4_gpu_tensor_contents(ref),H*D*4));
        ds4_gpu_tensor_free(qr);ds4_gpu_tensor_free(rr);ds4_gpu_tensor_free(ir);
    }
    assert(ds4_gpu_dsv41_dspark_rows(0)==1);g_ds41_levers.verify_batch_core=old;
    ds4_gpu_tensor *ts[]={q,raw,comp,ids,out,ref};for(unsigned i=0;i<6;i++)ds4_gpu_tensor_free(ts[i]);
    puts("PASS: indexed verify split attention every word equals eight independent chronological decode rows");
}


static void check_verify_low_rows(void) {
    enum { N=64,GROUPS=2,ROWS=8 };
    const unsigned widths[]={256,4096};
    for(unsigned shape=0;shape<2;shape++) {
        unsigned K=widths[shape];uint64_t bytes=(uint64_t)GROUPS*N*(K/32)*34;
        id<MTLBuffer> owned=ds4_metal_owned_wired_buffer(g_device,bytes,ds4_gpu_model_resource_options(),NULL);
        assert(owned && g_model_view_count<DS4_METAL_MAX_MODEL_VIEWS);
        unsigned char *map=owned.contents;
        for(uint64_t i=0;i<bytes;i+=34) {
            map[i]=0;map[i+1]=0x38;
            for(unsigned j=0;j<32;j++)map[i+2+j]=(unsigned char)((int)((i/34+j*7)%31)-15);
        }
        g_model_views[g_model_view_count++]=(ds4_gpu_model_view){.buffer=owned,.model_map=map,.model_size=bytes,.model_offset=0,.bytes=bytes};
        ds4_gpu_tensor *heads=ds4_gpu_tensor_alloc((uint64_t)ROWS*GROUPS*K*4),
            *out=ds4_gpu_tensor_alloc(ROWS*GROUPS*N*4),*ref=ds4_gpu_tensor_alloc(GROUPS*N*4);
        assert(heads&&out&&ref);
        float *in=ds4_gpu_tensor_contents(heads);for(unsigned i=0;i<ROWS*GROUPS*K;i++)in[i]=(float)((int)(i*17%733)-366)/512.f;
        float reference[ROWS*GROUPS*N],actual[ROWS*GROUPS*N];
        for(unsigned rows=2;rows<=ROWS;rows++) {
            for(unsigned r=0;r<rows;r++) {
                ds4_gpu_tensor *h=ds4_gpu_tensor_view(heads,(uint64_t)r*GROUPS*K*4,GROUPS*K*4);assert(h);
                assert(ds4_gpu_attention_output_low_q8_tensor(ref,map,bytes,0,K,N,GROUPS,h,0));
                assert(ds4_gpu_tensor_read(ref,0,reference+r*GROUPS*N,GROUPS*N*4));ds4_gpu_tensor_free(h);
            }
            g_ds41_levers.rows_mma=0;
            assert(ds4_gpu_attention_output_low_q8_rows_exact_tensor(out,map,bytes,0,K,N,GROUPS,0,GROUPS,heads,rows));
            assert(ds4_gpu_tensor_read(out,0,actual,rows*GROUPS*N*4));
            assert(!memcmp(reference,actual,rows*GROUPS*N*4));
            if(rows>=3) {
                g_ds41_levers.rows_mma=1;assert(ds4_gpu_dsv41_dspark_rows(1)==0);
                assert(ds4_gpu_attention_output_low_q8_rows_exact_tensor(out,map,bytes,0,K,N,GROUPS,0,GROUPS,heads,rows));
                assert(ds4_gpu_tensor_read(out,0,actual,rows*GROUPS*N*4));
                assert(ds4_gpu_dsv41_dspark_rows(0)==1);
                float scale=0;for(unsigned i=0;i<rows*GROUPS*N;i++)scale=fmaxf(scale,fabsf(reference[i]));
                for(unsigned i=0;i<rows*GROUPS*N;i++)assert(isfinite(actual[i])&&fabsf(reference[i]-actual[i])<=1.e-5f*scale);
            }
        }
        assert(!ds4_gpu_attention_output_low_q8_rows_exact_tensor(out,map,bytes,bytes,K,N,GROUPS,0,GROUPS,heads,ROWS));
        ds4_gpu_tensor_free(heads);ds4_gpu_tensor_free(out);ds4_gpu_tensor_free(ref);
    }
    g_ds41_levers.rows_mma=0;
    puts("PASS: attention low shared weight pass equals independent rows2-8, two K widths; matrix rows meet upstream float bound; bounds refuse");
}


static void check_verify_projection_tail(void) {
    enum { N=5120,MAXK=8192,MAXR=6 };
    ds41_levers_init_from_env();
    int op=g_ds41_levers.mtp_q8_pair6,os=g_ds41_levers.mtp_q8_stream6;
    g_ds41_levers.mtp_q8_pair6=0;g_ds41_levers.mtp_q8_stream6=0;
    for(unsigned shape=0;shape<2;shape++) {
        unsigned K=shape?2304:8192;uint64_t bytes=(uint64_t)N*K/32*34;
        id<MTLBuffer> owned=ds4_metal_owned_wired_buffer(g_device,bytes,ds4_gpu_model_resource_options(),NULL);
        assert(owned && g_model_view_count<DS4_METAL_MAX_MODEL_VIEWS);
        unsigned char *map=owned.contents;
        for(uint64_t i=0;i<bytes;i+=34) {
            map[i]=0;map[i+1]=0x38;for(unsigned j=0;j<32;j++)map[i+2+j]=(unsigned char)((int)((i/34+j*7)%31)-15);
        }
        g_model_views[g_model_view_count++]=(ds4_gpu_model_view){.buffer=owned,.model_map=map,.model_size=bytes,.model_offset=0,.bytes=bytes};
        ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(MAXR*K*4),*add=ds4_gpu_tensor_alloc(MAXR*N*4),*res=ds4_gpu_tensor_alloc(MAXR*N*4*4),
            *split=ds4_gpu_tensor_alloc(MAXR*24*4),*pr=ds4_gpu_tensor_alloc(MAXR*N*4),*pg=ds4_gpu_tensor_alloc(MAXR*N*4),
            *br=ds4_gpu_tensor_alloc(MAXR*N*4),*bg=ds4_gpu_tensor_alloc(MAXR*N*4),
            *hr=ds4_gpu_tensor_alloc(MAXR*N*4*4),*hg=ds4_gpu_tensor_alloc(MAXR*N*4*4);
        ds4_gpu_tensor *all[]={x,add,res,split,pr,pg,br,bg,hr,hg};for(unsigned i=0;i<10;i++)assert(all[i]);
        float *v=ds4_gpu_tensor_contents(x);for(unsigned i=0;i<MAXR*K;i++)v[i]=(float)((int)(i*17%733)-366)/512.f;
        v=ds4_gpu_tensor_contents(add);for(unsigned i=0;i<MAXR*N;i++)v[i]=(float)((int)(i*31%991)-495)/512.f;
        v=ds4_gpu_tensor_contents(res);for(unsigned i=0;i<MAXR*N*4;i++)v[i]=(float)((int)(i*13%997)-498)/512.f;
        v=ds4_gpu_tensor_contents(split);for(unsigned i=0;i<MAXR*24;i++)v[i]=(float)((int)(i%29)-14)/32.f;
        for(unsigned rows=2;rows<=6;rows+=4)for(unsigned mma=0;mma<2;mma++)for(unsigned plus=0;plus<2;plus++) {
            g_ds41_levers.rows_mma=mma;assert(ds4_gpu_dsv41_dspark_rows(1)==0);
            assert(ds4_gpu_matmul_q8_0_decode_rows_exact_impl(pr,map,bytes,0,K,N,x,rows,true));
            assert(ds4_gpu_dsv41_verify_epilogue_tensor(hr,br,plus?add:pr,plus?pr:NULL,res,split,rows));
            assert(ds4_gpu_dsv41_projection_hc_rows(hg,bg,pg,x,plus?add:NULL,res,split,map,bytes,0,K,N,rows));
            assert(ds4_gpu_dsv41_dspark_rows(0)==1);
            assert(!memcmp(ds4_gpu_tensor_contents(pr),ds4_gpu_tensor_contents(pg),rows*N*4));
            assert(!memcmp(ds4_gpu_tensor_contents(br),ds4_gpu_tensor_contents(bg),rows*N*4));
            assert(!memcmp(ds4_gpu_tensor_contents(hr),ds4_gpu_tensor_contents(hg),rows*N*4*4));
        }
        assert(!ds4_gpu_dsv41_projection_hc_rows(hg,bg,pg,x,NULL,res,split,map,bytes,bytes,K,N,MAXR));
        for(unsigned i=0;i<10;i++)ds4_gpu_tensor_free(all[i]);
    }
    g_ds41_levers.mtp_q8_pair6=op;g_ds41_levers.mtp_q8_stream6=os;g_ds41_levers.rows_mma=0;
    puts("PASS: projection/HC row join every projection/block/HC word equals retained chain, rows2/6, K2304/8192, scalar/matrix, add/no-add; bounds refuse");
}


static void check_scalar_upstream_index(void) {
    enum { MAXK=98305,D=128,H=32 };
    ds4_gpu_tensor *keys=ds4_gpu_tensor_alloc((uint64_t)MAXK*D*4),*view=ds4_gpu_tensor_alloc((uint64_t)MAXK*D*2),
        *q=ds4_gpu_tensor_alloc(H*D*4),*weights=ds4_gpu_tensor_alloc(H*4),
        *ref=ds4_gpu_tensor_alloc(MAXK*4),*got=ds4_gpu_tensor_alloc(MAXK*4);
    ds4_gpu_tensor *all[]={keys,view,q,weights,ref,got};for(unsigned i=0;i<6;i++)assert(all[i]);
    uint32_t *kv=ds4_gpu_tensor_contents(keys);
    for(unsigned i=0;i<MAXK*D;i++) { float f=(float)((int)(i*17%997)-498)/512.f;memcpy(kv+i,&f,4);kv[i]&=0xffff0000u; }
    kv[0]=0x80000000u;
    float *v=ds4_gpu_tensor_contents(q);for(unsigned i=0;i<H*D;i++)v[i]=(float)((int)(i*31%733)-366)/512.f;
    v=ds4_gpu_tensor_contents(weights);for(unsigned i=0;i<H;i++)v[i]=(float)(i%23+1)/32.f;
    assert(ds4_gpu_dsv41_index_keys_bf16(view,keys,0,MAXK,NULL));
    const unsigned cases[]={8192,8193,10239,10240,65535,65536,98303,98304,98305};
    for(unsigned c=0;c<sizeof(cases)/sizeof(*cases);c++) {
        unsigned n=cases[c];
        assert(ds4_gpu_dsv41_indexer_score_row_upstream(ref,q,weights,keys,NULL,n));
        assert(ds4_gpu_dsv41_indexer_score_row_upstream(got,q,weights,keys,view,n));
        assert(!memcmp(ds4_gpu_tensor_contents(ref),ds4_gpu_tensor_contents(got),n*4));
    }
    for(unsigned i=0;i<4;i++)assert(g_dsv41_upstream_index_dispatch[i]>0);
    const unsigned small[]={1,7,8191};
    for(unsigned c=0;c<3;c++) {
        unsigned n=small[c];
        assert(ds4_gpu_dsv41_indexer_score_one_tensor(ref,q,weights,keys,n,H,D,1.f/64.f,false));
        assert(ds4_gpu_dsv41_indexer_score_row_upstream(got,q,weights,keys,view,n));
        assert(!memcmp(ds4_gpu_tensor_contents(ref),ds4_gpu_tensor_contents(got),n*4));
    }
    assert(!ds4_gpu_dsv41_indexer_score_row_upstream(got,q,weights,keys,view,MAXK+1));
    assert(!ds4_gpu_dsv41_indexer_score_row_upstream(got,weights,weights,keys,view,8192));
    ds4_gpu_tensor *short_view=ds4_gpu_tensor_view(view,0,64);
    assert(short_view&&!ds4_gpu_dsv41_indexer_score_row_upstream(got,q,weights,keys,short_view,8192));
    ds4_gpu_tensor_free(short_view);
    for(unsigned i=0;i<6;i++)ds4_gpu_tensor_free(all[i]);
    assert(ds41_levers_set("index_schedule",0)&&ds41_levers_set("index_schedule",1)&&ds41_levers_set("index_schedule",2));
    assert(!ds41_levers_set("index_schedule",-1)&&!ds41_levers_set("index_schedule",3));
    assert(ds41_levers_set("index_schedule",0));
    puts("PASS: scalar upstream BF16/F32 scores every word at 8192/10240/65536/98304 thresholds and tails; signed zero; short retained path; all four dispatches witnessed; bounds and lever range");
}


static void check_query_rope_writer(void) {
    const uint64_t qw=1280ull/32*34*32768,hco=qw,hcs=20480ull*24*2,so=hco+hcs,bo=so+16,no=bo+96,bytes=no+2048;
    id<MTLBuffer> storage=ds4_metal_owned_wired_buffer(g_device,bytes,ds4_gpu_model_resource_options(),NULL);
    assert(storage&&g_model_view_count<DS4_METAL_MAX_MODEL_VIEWS);unsigned char *map=storage.contents;
    for(uint64_t i=0;i<qw;i+=34) {map[i]=0;map[i+1]=0x38;for(unsigned k=0;k<32;k++)map[i+2+k]=(unsigned char)((int)((i/34+k*7)%31)-15);}
    uint16_t *hw=(uint16_t *)(map+hco);for(unsigned i=0;i<hcs/2;i++)hw[i]=(uint16_t)((i%2?0x9c00:0x1c00));
    float *v=(float *)(map+so);for(unsigned i=0;i<4;i++)v[i]=.05f;
    v=(float *)(map+bo);for(unsigned i=0;i<24;i++)v[i]=(float)((int)(i%7)-3)/32.f;
    v=(float *)(map+no);for(unsigned i=0;i<512;i++)v[i]=(float)(i%17+1)/16.f;
    g_model_views[g_model_view_count++]=(ds4_gpu_model_view){.buffer=storage,.model_map=map,.model_size=bytes,.model_offset=0,.bytes=bytes};
    ds4_gpu_tensor *q[2],*mix[2],*split[2],*kv[2],*ring[2],*counter[2];
    for(unsigned i=0;i<2;i++) {q[i]=ds4_gpu_tensor_alloc(32768*4);mix[i]=ds4_gpu_tensor_alloc(96);split[i]=ds4_gpu_tensor_alloc(96);kv[i]=ds4_gpu_tensor_alloc(512*4);ring[i]=ds4_gpu_tensor_alloc(128*512*4);counter[i]=ds4_gpu_tensor_alloc(4);assert(q[i]&&mix[i]&&split[i]&&kv[i]&&ring[i]&&counter[i]);}
    ds4_gpu_tensor *input=ds4_gpu_tensor_alloc(1280*4),*res=ds4_gpu_tensor_alloc(20480*4);assert(input&&res);
    v=ds4_gpu_tensor_contents(input);for(unsigned i=0;i<1280;i++)v[i]=(float)((int)(i*17%991)-495)/256.f;
    v=ds4_gpu_tensor_contents(res);for(unsigned i=0;i<20480;i++)v[i]=(float)((int)(i*31%997)-498)/256.f;
    const unsigned positions[]={0,8192,131072,1048575};
    uint32_t zero=0;
    assert(ds4_gpu_dsv41_query_rope_available());
    for(unsigned cp=0;cp<2;cp++)for(unsigned p=0;p<4;p++) {
        unsigned pos=positions[p];
        for(unsigned arm=0;arm<2;arm++) {
            v=ds4_gpu_tensor_contents(kv[arm]);for(unsigned i=0;i<512;i++)v[i]=(float)((int)(i*13%997)-498)/256.f;
            memset(ds4_gpu_tensor_contents(ring[arm]),0,128*512*4);assert(ds4_gpu_tensor_write(counter[arm],0,&zero,4));
            assert((arm?ds4_gpu_dsv41_query_hc_kv_rope_tensor:ds4_gpu_dsv41_query_hc_kv_tensor)(q[arm],mix[arm],split[arm],counter[arm],res,input,map,bytes,hco,so,bo,0,20,1.e-6f,1.e-6f,kv[arm],ring[arm],no,pos,cp));
            if(!arm)assert(ds4_gpu_dsv41_rope(q[arm],512,64,1,pos,cp,false));
        }
        const ds4_gpu_tensor *a[]={q[0],mix[0],split[0],kv[0],ring[0]},*b[]={q[1],mix[1],split[1],kv[1],ring[1]};
        const unsigned sizes[]={32768*4,96,96,512*4,128*512*4};
        for(unsigned j=0;j<5;j++) {assert(ds4_gpu_synchronize());assert(!memcmp(ds4_gpu_tensor_contents(a[j]),ds4_gpu_tensor_contents(b[j]),sizes[j]));}
    }
    assert(!ds4_gpu_dsv41_query_hc_kv_rope_tensor(q[1],mix[1],split[1],counter[1],res,input,map,bytes,hco,so,bo,0,20,1.e-6f,1.e-6f,kv[1],ring[1],no,1048576,false));
    assert(!ds4_gpu_dsv41_query_hc_kv_rope_tensor(q[1],mix[1],split[1],counter[1],res,input,map,bytes,hco,so,bo,0,20,1.e-6f,1.e-6f,NULL,NULL,no,8192,false));
    for(unsigned i=0;i<2;i++) {ds4_gpu_tensor_free(q[i]);ds4_gpu_tensor_free(mix[i]);ds4_gpu_tensor_free(split[i]);ds4_gpu_tensor_free(kv[i]);ds4_gpu_tensor_free(ring[i]);ds4_gpu_tensor_free(counter[i]);}
    ds4_gpu_tensor_free(input);ds4_gpu_tensor_free(res);
    puts("PASS: fused query-store RoPE all query/mix/split/KV/ring words equal retained producer+rotation at 0/8K/131K/1M-1, both frequencies; position refusal");
}


static void check_short_topk(void) {
    enum { N=16384,K=512 };
    ds4_gpu_tensor *scores=ds4_gpu_tensor_alloc(2*N*4),*ref=ds4_gpu_tensor_alloc(4096*4),*got=ds4_gpu_tensor_alloc(4096*4);
    assert(scores&&ref&&got);float *sv=ds4_gpu_tensor_contents(scores);
    const unsigned sizes[]={512,513,1023,1024,1025,4095,8192,8193,16383,16384};
    for(unsigned pattern=0;pattern<4;pattern++) {
        for(unsigned i=0;i<2*N;i++) {
            float v=(float)((int)((i*8191u+17u)%32749u)-16374)/128.f;
            if(pattern==1)v=(float)(i%13);
            if(pattern==2)v=(i&1)?0.f:-0.f;
            if(pattern==3) { if(i%5==0)v=-INFINITY;if(i%997==0)v=INFINITY; }
            sv[i]=v;
        }
        for(unsigned c=0;c<sizeof(sizes)/sizeof(*sizes);c++) {
            unsigned n=sizes[c];
            g_ds41_levers.topk_short=0;assert(ds4_gpu_dsv41_indexer_topk_radix(ref,scores,n,false));
            g_ds41_levers.topk_short=1;assert(ds4_gpu_dsv41_indexer_topk_radix(got,scores,n,false));
            assert(!memcmp(ds4_gpu_tensor_contents(ref),ds4_gpu_tensor_contents(got),K*4));
        }
        for(unsigned arm=0;arm<2;arm++) {
            g_ds41_levers.topk_short=arm;
            assert(ds4_gpu_indexer_topk_tensor(arm?got:ref,scores,8192,1,2048));
        }
        assert(!memcmp(ds4_gpu_tensor_contents(ref),ds4_gpu_tensor_contents(got),2048*4));
        for(unsigned arm=0;arm<2;arm++) {
            g_ds41_levers.topk_short=arm;
            assert(ds4_gpu_dsv41_indexer_topk_batch(arm?got:ref,scores,4096,2,4096,2));
        }
        assert(!memcmp(ds4_gpu_tensor_contents(ref),ds4_gpu_tensor_contents(got),2*K*4));
    }
    for(unsigned i=0;i<3;i++)assert(g_dsv41_short_topk_dispatches[i]>0);
    assert(!ds4_gpu_dsv41_indexer_topk_radix(got,scores,511,false));
    g_ds41_levers.topk_short=0;ds4_gpu_tensor_free(scores);ds4_gpu_tensor_free(ref);ds4_gpu_tensor_free(got);
    puts("PASS: short top512 every selected index exact at512/1024/4096/8192/16384 boundaries and tails, unique/ties/signedzero/infinite; all3dispatches witnessed; longradix/blocktop2048/causalbatch retain original words; bounds");
}


static void check_literal_hc_join(void) {
    enum { N=5120 };
    ds4_gpu_tensor *in=ds4_gpu_tensor_alloc(N*4),*add=ds4_gpu_tensor_alloc(N*4),*res=ds4_gpu_tensor_alloc(4*N*4),*split=ds4_gpu_tensor_alloc(96),
        *hc[2],*block[2];assert(in&&add&&res&&split);
    for(unsigned j=0;j<2;j++) {hc[j]=ds4_gpu_tensor_alloc(4*N*4);block[j]=ds4_gpu_tensor_alloc(N*4);assert(hc[j]&&block[j]);}
    float *v=ds4_gpu_tensor_contents(in);for(unsigned i=0;i<N;i++)v[i]=(float)((int)(i*17%997)-498)/257.f;
    v=ds4_gpu_tensor_contents(add);for(unsigned i=0;i<N;i++)v[i]=(float)((int)(i*31%991)-495)/513.f;
    v=ds4_gpu_tensor_contents(split);for(unsigned i=0;i<24;i++)v[i]=(float)((int)(i%29)-14)/33.f;
    for(unsigned plus=0;plus<2;plus++)for(unsigned alias=0;alias<2;alias++) {
        for(unsigned arm=0;arm<2;arm++) {
            v=ds4_gpu_tensor_contents(res);for(unsigned i=0;i<4*N;i++)v[i]=(float)((int)(i*13%733)-366)/511.f;
            g_ds41_levers.hc_join_static=arm;
            assert(ds4_gpu_dsv41_hc_round_expand4_tensor(alias?res:hc[arm],block[arm],in,plus?add:NULL,res,split,N,4));
            if(alias)assert(ds4_gpu_tensor_read(res,0,ds4_gpu_tensor_contents(hc[arm]),4*N*4));
        }
        assert(!memcmp(ds4_gpu_tensor_contents(hc[0]),ds4_gpu_tensor_contents(hc[1]),4*N*4));
        assert(!memcmp(ds4_gpu_tensor_contents(block[0]),ds4_gpu_tensor_contents(block[1]),N*4));
    }
    assert(g_dsv41_hc_literal_dispatches>0);g_ds41_levers.hc_join_static=0;
    for(unsigned i=0;i<2;i++){ds4_gpu_tensor_free(hc[i]);ds4_gpu_tensor_free(block[i]);}
    ds4_gpu_tensor_free(in);ds4_gpu_tensor_free(add);ds4_gpu_tensor_free(res);ds4_gpu_tensor_free(split);
    puts("PASS: literal HC FFN join every HC/block word exact, add/noadd, separate/inplace residual; actual static dispatch witnessed");
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
        ds41_levers_init_from_env();
        check_literal_hc_join();
        check_short_topk();
        check_query_rope_writer();
        check_scalar_upstream_index();
        check_verify_projection_tail();
        check_verify_low_rows();
        check_verify_attention_rows();
        check_verify_indexed();
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
