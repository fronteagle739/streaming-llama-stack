/* Small tensor-kernel oracle, not a model cross-quant application test.
 * Each quantized fixture is decoded independently on CPU and compared with
 * (W - sum_k s_k r_k(r_k^T W))x in double precision. No GGUF is created. */
#include "../ds4_gpu.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {INPUT=256,OUTPUT=5120,COUNT=2};
static unsigned char *sources[6];
static unsigned source_count;

static void put16(unsigned char *p,uint16_t value) {p[0]=(unsigned char)value;p[1]=(unsigned char)(value>>8);}

static size_t fixture(unsigned char *bytes,float *weight,unsigned type,int variant) {
    const size_t row_bytes=type==0?INPUT*4:type==8?(INPUT/32)*34:84;
    for(unsigned row=0;row<OUTPUT;row++) {
        unsigned char *data=bytes+row*row_bytes;
        if(type==0) {
            for(unsigned col=0;col<INPUT;col++) {
                float v=.001f*(float)((int)((row*7+col*3+variant)%37)-18);
                memcpy(data+col*4,&v,4);weight[row*INPUT+col]=v;
            }
        } else if(type==8) {
            for(unsigned block=0;block<INPUT/32;block++) {
                put16(data+block*34,0x2800); /* 2^-5 */
                for(unsigned j=0;j<32;j++) {
                    int q=(int)((row*3+block*11+j+variant)%31)-15;
                    data[block*34+2+j]=(unsigned char)(int8_t)q;
                    weight[row*INPUT+block*32+j]=(float)q/32.f;
                }
            }
        } else {
            for(unsigned j=0;j<16;j++)data[j]=(unsigned char)(((j+row+variant)%15+1)|(((j+row)%5)<<4));
            for(unsigned j=0;j<64;j++)data[16+j]=(unsigned char)((j*17+row*13+variant)%256);
            put16(data+80,0x2400);put16(data+82,0x2000); /* 2^-6, 2^-7 */
            for(unsigned chunk=0;chunk<2;chunk++)for(unsigned group=0;group<4;group++)
                for(unsigned pair=0;pair<2;pair++)for(unsigned j=0;j<16;j++) {
                    unsigned scale=data[chunk*8+group*2+pair];
                    unsigned q=(data[16+chunk*32+pair*16+j]>>(2*group))&3;
                    weight[row*INPUT+chunk*128+group*32+pair*16+j]=
                        (float)(scale&15)*(float)q/64.f-(float)(scale>>4)/128.f;
                }
        }
    }
    return OUTPUT*row_bytes;
}

static double check(unsigned type,int variant,ds4_gpu_tensor *xt,ds4_gpu_tensor *pt,ds4_gpu_tensor *out,
        const float *x,const float *pack) {
    const size_t maximum=(size_t)INPUT*OUTPUT*4;
    unsigned char *source=malloc(maximum);float *weight=malloc(maximum),*got=malloc(OUTPUT*sizeof(float));
    assert(source && weight && got);
    assert(source_count<6);sources[source_count++]=source;
    size_t bytes=fixture(source,weight,type,variant);
    assert(ds4_gpu_set_model_map(source,bytes));
    assert(ds4_gpu_begin_commands());
    assert(ds4_gpu_direction_matmul(out,source,bytes,0,type,INPUT,OUTPUT,xt,pt,0,0,COUNT));
    assert(ds4_gpu_end_commands() && ds4_gpu_synchronize());
    assert(ds4_gpu_tensor_read(out,0,got,OUTPUT*sizeof(float)));
    float cached[OUTPUT];uint64_t fusions=0,hits=0,resident=0;
    assert(ds4_gpu_begin_commands());
    assert(ds4_gpu_direction_matmul(out,source,bytes,0,type,INPUT,OUTPUT,xt,pt,0,0,COUNT));
    assert(ds4_gpu_end_commands() && ds4_gpu_synchronize());
    assert(ds4_gpu_tensor_read(out,0,cached,sizeof(cached)));
    ds4_gpu_direction_fused_stats(&fusions,&hits,&resident);
    assert(fusions==1 && hits==1 && resident==(uint64_t)INPUT*OUTPUT*sizeof(float));
    assert(memcmp(got,cached,sizeof(cached))==0);
    double values[OUTPUT],dots[COUNT]={0},worst=0;
    for(unsigned row=0;row<OUTPUT;row++) {
        double value=0;for(unsigned col=0;col<INPUT;col++)value+=(double)weight[row*INPUT+col]*x[col];
        values[row]=value;
        for(unsigned k=0;k<COUNT;k++)dots[k]+=value*pack[k*(OUTPUT+1)+1+row];
    }
    for(unsigned row=0;row<OUTPUT;row++) {
        double expected=values[row];
        for(unsigned k=0;k<COUNT;k++)expected-=(double)pack[k*(OUTPUT+1)]*pack[k*(OUTPUT+1)+1+row]*dots[k];
        double error=fabs((double)got[row]-expected);if(error>worst)worst=error;
        assert(isfinite(got[row]) && error<2e-4*(1+fabs(expected)));
    }
    printf("target fusion type=%u fixture=%d worst_abs=%.9g PASS\n",type,variant,worst);
    ds4_gpu_direction_fused_clear();
    /* The CUDA mapped-weight cache owns a source identity until cleanup. */
    free(weight);free(got);return worst;
}

/* An owned Python driver supplies three actual 1017 writer blocks and compares
 * this output with its independent CPU-double dequantization oracle. No model
 * inference or other quant is loaded by this bounded transport fixture. */
static int real_writer(const char *packet_path,const char *output_path,int mixed) {
    FILE *file=fopen(packet_path,"rb");assert(file);
    char magic[4];uint32_t version,input,output,type;uint64_t bytes;
    assert(fread(magic,1,4,file)==4 && memcmp(magic,"D4FM",4)==0);
    assert(fread(&version,4,1,file)==1 && version==1);
    assert(fread(&input,4,1,file)==1 && input>0 && input<=32768);
    assert(fread(&output,4,1,file)==1 && output==OUTPUT);
    assert(fread(&type,4,1,file)==1 && (type==0 || type==1 || type==8 || type==10 || type==12 || type==30 || type==39 || type==0x10000u+39u));
    assert(fread(&bytes,8,1,file)==1 && bytes>0 && bytes<=200ull*1024*1024);
    const size_t pack_bytes=COUNT*(OUTPUT+1)*sizeof(float);
    float *x=malloc(input*sizeof(float)),*pack=malloc(pack_bytes),got[OUTPUT],again[OUTPUT];
    unsigned char *source=malloc((size_t)bytes);assert(x&&pack&&source);
    assert(fread(x,sizeof(float),input,file)==input);
    assert(fread(pack,1,pack_bytes,file)==pack_bytes && fread(source,1,(size_t)bytes,file)==bytes);
    float cpu_base[OUTPUT];
    if(mixed){assert(type==0x10000u+39u && input==2304u);assert(fread(cpu_base,sizeof(float),OUTPUT,file)==OUTPUT);}
    assert(fgetc(file)==EOF);fclose(file);
    for(unsigned i=0;i<input;i++)assert(isfinite(x[i]));
    for(unsigned i=0;i<pack_bytes/sizeof(float);i++)assert(isfinite(pack[i]));
    assert(ds4_gpu_init() && ds4_gpu_set_model_map(source,bytes));
    ds4_gpu_tensor *xt=ds4_gpu_tensor_alloc(input*sizeof(float)),*pt=ds4_gpu_tensor_alloc(pack_bytes);
    ds4_gpu_tensor *out=ds4_gpu_tensor_alloc(sizeof(got));assert(xt&&pt&&out);
    ds4_gpu_tensor *cpu=mixed?ds4_gpu_tensor_alloc(sizeof(cpu_base)):NULL;
    if(mixed){assert(cpu);assert(ds4_gpu_tensor_write(cpu,0,cpu_base,sizeof(cpu_base)));}
    assert(ds4_gpu_tensor_write(xt,0,x,input*sizeof(float)) && ds4_gpu_tensor_write(pt,0,pack,pack_bytes));
    for(unsigned pass=0;pass<2;pass++) {
        assert(ds4_gpu_begin_commands());
        assert(mixed?ds4_gpu_direction_packed_mixed_control(out,source,bytes,0,xt,pt,cpu,0,0,COUNT):
            ds4_gpu_direction_matmul(out,source,bytes,0,type,input,output,xt,pt,0,0,COUNT));
        assert(ds4_gpu_end_commands() && ds4_gpu_synchronize());
        assert(ds4_gpu_tensor_read(out,0,pass?again:got,sizeof(got)));
    }
    uint64_t fusions=0,hits=0,resident=0;
    ds4_gpu_direction_fused_stats(&fusions,&hits,&resident);
    assert(fusions==1 && hits==1 && resident==(type==0x10000u+39u?0u:(uint64_t)input*output*sizeof(float)));
    assert(memcmp(got,again,sizeof(got))==0);
    for(unsigned i=0;i<OUTPUT;i++)assert(isfinite(got[i]));
    file=fopen(output_path,"wbx");assert(file);
    assert(fwrite(got,1,sizeof(got),file)==sizeof(got));fclose(file);
    printf("real writer input=%u output=%u type=%u mixed=%d fusions=%llu hits=%llu resident=%llu PASS\n",
        input,output,type,mixed,(unsigned long long)fusions,(unsigned long long)hits,(unsigned long long)resident);
    ds4_gpu_direction_fused_clear();
    ds4_gpu_tensor_free(xt);ds4_gpu_tensor_free(pt);ds4_gpu_tensor_free(out);ds4_gpu_tensor_free(cpu);ds4_gpu_cleanup();
    free(x);free(pack);free(source);return 0;
}

int main(int argc,char **argv) {
    if(argc==4 && strcmp(argv[1],"--real")==0)return real_writer(argv[2],argv[3],0);
    if(argc==4 && strcmp(argv[1],"--real-mixed")==0)return real_writer(argv[2],argv[3],1);
    assert(argc==1);
    float x[INPUT],pack[2*COUNT*(OUTPUT+1)]={0};
    for(unsigned col=0;col<INPUT;col++)x[col]=.02f*(float)((int)(col%17)-8);
    for(unsigned k=0;k<COUNT;k++) {
        double norm=0;
        for(unsigned row=0;row<OUTPUT;row++) {
            float v=(float)sin((double)row*.13+k*.4);
            pack[k*(OUTPUT+1)+1+row]=v;norm+=(double)v*v;
        }
        for(unsigned row=0;row<OUTPUT;row++)pack[k*(OUTPUT+1)+1+row]/=(float)sqrt(norm);
        pack[k*(OUTPUT+1)]=k?.7f:.5f;
    }
    assert(ds4_gpu_init());
    ds4_gpu_tensor *xt=ds4_gpu_tensor_alloc(sizeof(x)),*pt=ds4_gpu_tensor_alloc(sizeof(pack));
    ds4_gpu_tensor *out=ds4_gpu_tensor_alloc(OUTPUT*sizeof(float));assert(xt&&pt&&out);
    assert(ds4_gpu_tensor_write(xt,0,x,sizeof(x)) && ds4_gpu_tensor_write(pt,0,pack,sizeof(pack)));
    const unsigned types[]={0,8,10};
    for(unsigned i=0;i<3;i++)for(int variant=0;variant<2;variant++)check(types[i],variant,xt,pt,out,x,pack);
    assert(!ds4_gpu_direction_matmul(out,x,sizeof(x),0,8,INPUT,OUTPUT,xt,pt,0,0,COUNT));
    assert(!ds4_gpu_direction_matmul(out,x,sizeof(x),0,99,INPUT,OUTPUT,xt,pt,0,0,COUNT));
    ds4_gpu_direction_fused_clear();
    ds4_gpu_tensor_free(xt);ds4_gpu_tensor_free(pt);ds4_gpu_tensor_free(out);ds4_gpu_cleanup();
    for(unsigned i=0;i<source_count;i++)free(sources[i]);
    puts("simultaneous nonorthogonal target-weight fusion and bounds PASS");return 0;
}
