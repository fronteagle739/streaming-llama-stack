// Compare the admitted size-only plan with a second-reserve control on a real
// CPU backend. Exercise views, branched last-reader reuse, size changes and
// repeated buffer types, without a model or GPU allocation.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" bool ggml_gallocr_materialize_reserved(ggml_gallocr_t,ggml_cgraph *);

static void check(bool ok,const char *message){if(!ok){std::fprintf(stderr,"FAIL %s\n",message);std::exit(1);}}

static std::vector<float> run(ggml_backend_t backend,ggml_gallocr_t alloc,int rows,bool materialize){
    auto *ctx=ggml_init({1<<20,nullptr,true});check(ctx,"metadata");
    auto *x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,64,rows);
    auto *bias=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,64);
    ggml_set_input(x);ggml_set_input(bias);
    auto *sum=ggml_add(ctx,x,bias);
    auto *left=ggml_sqr(ctx,sum);
    auto *right=ggml_scale(ctx,sum,0.5f);
    auto *out=ggml_add(ctx,left,right);ggml_set_output(out);
    // A strided view keeps its parent live across the branch.
    auto *view=ggml_view_2d(ctx,x,32,rows,x->nb[1],16*sizeof(float));
    auto *reduced=ggml_sum(ctx,view);ggml_set_output(reduced);
    auto *graph=ggml_new_graph_custom(ctx,128,false);
    ggml_build_forward_expand(graph,out);ggml_build_forward_expand(graph,reduced);
    auto *other=ggml_new_graph_custom(ctx,128,false);
    check(!ggml_gallocr_materialize_reserved(alloc,other),"foreign graph must be refused");
    size_t planned[2]={};ggml_gallocr_reserve_n_size(alloc,graph,nullptr,nullptr,planned);
    check(planned[0]>0,"nonempty plan");
    check(!ggml_gallocr_materialize_reserved(alloc,other),"foreign graph after plan must be refused");
    ggml_gallocr_clear_reserved_tensors(alloc);
    check(!ggml_gallocr_materialize_reserved(alloc,graph),"clearing ranges invalidates size plan");
    ggml_gallocr_reserve_n_size(alloc,graph,nullptr,nullptr,planned);
    check(ggml_gallocr_set_reserved_tensors(alloc,nullptr,0),"empty reservation update");
    check(!ggml_gallocr_materialize_reserved(alloc,graph),"setting ranges invalidates size plan");
    ggml_gallocr_reserve_n_size(alloc,graph,nullptr,nullptr,planned);
    check(!ggml_gallocr_set_reserved_tensors(alloc,nullptr,1),"invalid reservation update refused");
    check(!ggml_gallocr_materialize_reserved(alloc,graph),"failed reservation update invalidates size plan");
    ggml_gallocr_reserve_n_size(alloc,graph,nullptr,nullptr,planned);
    check(materialize?ggml_gallocr_materialize_reserved(alloc,graph):ggml_gallocr_reserve(alloc,graph),"materialize/reserve");
    check(!ggml_gallocr_materialize_reserved(alloc,graph),"consumed or ordinary-reserve plan refused");
    check(ggml_gallocr_alloc_graph(alloc,graph),"initialize planned tensors");
    check(!ggml_gallocr_materialize_reserved(alloc,graph),"allocation invalidates size plan");
    check(ggml_gallocr_get_buffer_size(alloc,0)>=planned[0],"materialized buffer covers plan");
    std::vector<float> input(64*rows),b(64),result(64*rows+1);
    for(size_t i=0;i<input.size();++i)input[i]=float(int(i%31)-15)/16;
    for(size_t i=0;i<b.size();++i)b[i]=float(i)/64;
    ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
    ggml_backend_tensor_set(bias,b.data(),0,b.size()*4);
    check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"compute");
    ggml_backend_tensor_get(out,result.data(),0,(result.size()-1)*4);
    ggml_backend_tensor_get(reduced,&result.back(),0,4);
    ggml_free(ctx);return result;
}

int main(){
    auto *backend=ggml_backend_cpu_init();check(backend,"CPU backend");ggml_backend_cpu_set_n_threads(backend,2);
    auto buft=ggml_backend_get_default_buffer_type(backend);
    for(bool duplicated:{false,true}){
        ggml_backend_buffer_type_t types[2]={buft,buft};
        auto *control=ggml_gallocr_new_n(types,duplicated?2:1);
        auto *candidate=ggml_gallocr_new_n(types,duplicated?2:1);
        for(int rows:{1,127,17,1024,31}){
            auto a=run(backend,control,rows,false),b=run(backend,candidate,rows,true);
            check(a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*4),"all output bytes match");
        }
        ggml_gallocr_free(control);ggml_gallocr_free(candidate);
    }
    ggml_backend_free(backend);std::puts("PASS exact size-plan materialization, views, branches, growth/shrink and repeated buffer types");return 0;
}
