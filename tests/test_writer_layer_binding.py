"""Execute the real decode/HC host functions with a graph-replay seam."""
from pathlib import Path
import os
import shutil
import subprocess


def function(source, name):
    begin=source.index('static bool '+name+'(')
    brace=source.index('{',begin)
    depth=1;end=brace+1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[begin:end]


def test_actual_decode_keeps_layer_binding_when_hc_replays(tmp_path):
    compiler=shutil.which('clang++') or shutil.which('c++')
    assert compiler, 'C++ compiler required for actual host replay contract'
    source=(Path(__file__).resolve().parents[1]/'engines/ds4-win/ds4.c').read_text()
    body=r'''
#include <cassert>
#include <cstdint>
#include <cstdlib>
struct ds41_gpu_graph {
    int tp_world=1;bool streaming=false,quality=false,imatrix=false;
    unsigned image_count=0,direction_layer=999;
    void *residual=nullptr,*after_attn=nullptr,*ffn_split=nullptr,*attn_split=nullptr,*block=nullptr;
};
struct ds4_model {const void *map=nullptr;};
struct ds4_layer_weights {};
struct ds4_decode_graph_key {unsigned il,island,variant;void *cur_hc,*after_attn_hc,*after_ffn_hc,*attn_norm;};
static struct {bool active=false;}g_expert_profile;
static struct {const char *prefix=nullptr;}debug;
static auto metal_graph_debug_get_config(){return &debug;}
static unsigned g_ds41_hc_replays=0,g_ds41_hc_calls=0;
static double g_ds41_hc_encode_s=0;
static bool graphs_on=true,captured[40]={};
static unsigned active_graph=0,attention_calls=0;
static double now_sec(){return 1.;}
static bool ds41_hc_graph_profile_on(){return true;}
static bool ds41_single_hc_graphs_on(){return graphs_on;}
static bool ds4_gpu_decode_graphs_supported(){return true;}
static bool ds41_engram_layer(unsigned layer){return layer==1||layer==14;}
static int ds4_gpu_decode_graph_begin(const ds4_decode_graph_key*k){active_graph=k->il;return captured[k->il]?1:0;}
static int ds4_gpu_decode_graph_end(const ds4_decode_graph_key*){captured[active_graph]=true;return 0;}
static void ds4_gpu_decode_graph_abort(const ds4_decode_graph_key*){assert(false);}
static bool ds41_graph_before_attention(ds41_gpu_graph*g,const ds4_model*,const ds4_layer_weights*,unsigned layer){g->direction_layer=layer;return true;}
static bool ds41_attention(ds41_gpu_graph*g,const ds4_model*,const ds4_layer_weights*,unsigned layer,bool){++attention_calls;return g->direction_layer==layer;}
static bool ds41_hc_mix(ds41_gpu_graph*,const ds4_model*,const ds4_layer_weights*,bool,void*){return true;}
static bool ds41_moe(ds41_gpu_graph*g,const ds4_model*,const ds4_layer_weights*,unsigned layer,unsigned){return g->direction_layer==layer;}
static bool ds41_graph_after_moe(ds41_gpu_graph*,const ds4_model*,const ds4_layer_weights*,unsigned,bool){return true;}
static bool ds41_graph_layer(ds41_gpu_graph*g,const ds4_model*m,const ds4_layer_weights*l,unsigned layer,int){return ds41_graph_before_attention(g,m,l,layer)&&ds41_attention(g,m,l,layer,false);}
static bool ds41_decode_island(ds41_gpu_graph*,const ds4_model*,const ds4_layer_weights*,unsigned,unsigned){return true;}
static bool ds41_sum_partial(ds41_gpu_graph*,void*,unsigned,unsigned){return true;}
static bool ds41_bf16(void*,unsigned){return true;}
static bool ds41_moe_finish(ds41_gpu_graph*,unsigned){return true;}
#define DS41_MARK(x) true
#define DS4_N_LAYER 40
#define DS4_N_EMBD 5120
#define DS4_TP_GATE_ATTN 0
'''
    body+='\n'+function(source,'ds41_single_hc_island')+'\n'+function(source,'ds41_graph_decode_layer')
    body+=r'''
int main(){
    ds41_gpu_graph g;ds4_model m;ds4_layer_weights l;
    // First token captures, later tokens replay. Engram layers still take the
    // ordinary path, reproducing the stale layer1/layer14 hardware trace.
    for(unsigned token=0;token<3;++token)for(unsigned layer=0;layer<40;++layer)
        if(!ds41_graph_decode_layer(&g,&m,&l,layer,7))return 3;
    if(attention_calls!=120||g_ds41_hc_replays!=76||g_ds41_hc_calls!=114)return 4;
    graphs_on=false;
    for(unsigned layer=0;layer<40;++layer)if(!ds41_graph_decode_layer(&g,&m,&l,layer,7))return 5;
    graphs_on=true;g.tp_world=2;
    for(unsigned layer=0;layer<40;++layer)if(!ds41_graph_decode_layer(&g,&m,&l,layer,7))return 6;
}
'''
    path=tmp_path/'writer-layer.cpp';path.write_text(body)
    output=Path(os.environ.get('TMPDIR',str(tmp_path)))/'directions-writer-capture-test'
    subprocess.run([compiler,'-std=c++20','-O1',str(path),'-o',str(output)],check=True,timeout=60)
    subprocess.run([str(output)],check=True,timeout=10)
