#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

// A dedicated producer owns this immutable plan. Capture adds consumers to the
// ordinary graph; it never changes the graph's arithmetic or carry precision.
namespace dsv41_writer_capture {
inline void check(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
struct plan {
    uint32_t tokens=0,groups=0;
    bool active=false,evaluate=false;
    std::vector<uint32_t> positions,group,counts;
    std::vector<float> output;
    std::vector<float> final_residual,final_pre,evaluated;
    std::vector<uint8_t> final_seen;
    std::vector<uint8_t> seen;
    void configure(uint32_t n,uint32_t g,std::vector<uint32_t> p,std::vector<uint32_t> ids){
        check(!active&&n&&n<=2048&&g&&g<=16&&!p.empty()&&p.size()<=2048&&p.size()==ids.size(),"Invalid writer capture geometry");
        tokens=n;groups=g;positions=std::move(p);group=std::move(ids);counts.assign(g,0);
        for(size_t i=0;i<positions.size();++i){
            check(positions[i]<n&&group[i]<g&&(!i||positions[i]>positions[i-1]),"Invalid writer capture position/group");++counts[group[i]];
        }
        check(std::all_of(counts.begin(),counts.end(),[](auto c){return c!=0;}),"Empty writer capture group");
        output.assign(size_t(g)*40*6*5120,0.f);seen.assign(size_t(40)*positions.size()*6,0);active=true;
    }
    void enable_evaluate(){
        check(groups==positions.size(),"Evaluation requires raw positions, not grouped means");evaluate=true;
        final_residual.assign(size_t(groups)*4*5120,0);final_pre.assign(size_t(groups)*4,0);final_seen.assign(size_t(groups)*2,0);
        evaluated.assign(size_t(groups)*(40*6*5120+5120+129280),0);
    }
    void finish(){check(active&&std::all_of(seen.begin(),seen.end(),[](auto s){return s==1;}),"Writer capture did not visit every selected layer/site exactly once");
        if(evaluate)check(std::all_of(final_seen.begin(),final_seen.end(),[](auto s){return s==1;}),"Evaluation lost final residual/pre coverage");active=false;}
};
inline plan current;

struct graph_capture {
    struct row {uint32_t site,offset,count,width;ggml_tensor *weights,*value;};
    std::vector<row> rows;
    uint32_t attention_offset=0,ffn_offset=0;
    void add(ggml_context *ctx,ggml_cgraph *graph,ggml_tensor *value,uint32_t site,uint32_t offset,uint32_t count,bool round){
        if(!current.active)return;
        const uint32_t width=site==0||site==6?4*5120:site==10?4:5120;
        check(value&&ggml_nelements(value)==int64_t(width)*count,"Writer capture graph shape differs");
        auto *flat=ggml_reshape_2d(ctx,value,width,count);
        if(round)flat=ggml_cast(ctx,ggml_cast(ctx,flat,GGML_TYPE_BF16),GGML_TYPE_F32);
        else if(flat->type!=GGML_TYPE_F32)flat=ggml_cast(ctx,flat,GGML_TYPE_F32);
        auto *weights=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,count,current.groups);ggml_set_input(weights);
        auto *transposed=ggml_cont(ctx,ggml_transpose(ctx,flat));
        auto *result=ggml_mul_mat(ctx,transposed,weights);ggml_mul_mat_set_prec(result,GGML_PREC_F32);
        ggml_set_output(result);ggml_build_forward_expand(graph,result);
        rows.push_back({site,offset,count,width,weights,result});
    }
    void observe(ggml_context *ctx,ggml_cgraph *graph,ggml_tensor *t,const char *name){
        if(!current.active)return;
        if(!std::strcmp(name,"dsv41_writer_attention")){
            auto count=uint32_t(t->ne[1]);add(ctx,graph,t,4,attention_offset,count,true);attention_offset+=count;
        }else if(!std::strcmp(name,"dsv41_writer_ffn")){
            auto count=uint32_t(t->ne[1]);add(ctx,graph,t,5,ffn_offset,count,true);ffn_offset+=count;
        }
    }
    void upload(uint32_t start)const{
        if(!current.active)return;
        for(const auto &r:rows){
            std::vector<float> weights(size_t(r.count)*current.groups,0.f);
            for(size_t i=0;i<current.positions.size();++i){const auto p=current.positions[i];
                if(p>=start+r.offset&&p<start+r.offset+r.count)
                    weights[size_t(current.group[i])*r.count+p-start-r.offset]=1.f/float(current.counts[current.group[i]]);
            }
            ggml_backend_tensor_set(r.weights,weights.data(),0,weights.size()*sizeof(float));
        }
    }
    void collect(uint32_t layer,uint32_t start)const{
        if(!current.active)return;check(layer<40,"Invalid writer capture layer");
        for(const auto &r:rows){
            std::vector<float> values(size_t(r.width)*current.groups);
            ggml_backend_tensor_get(r.value,values.data(),0,values.size()*sizeof(float));
            for(size_t i=0;i<current.positions.size();++i)if(current.positions[i]>=start+r.offset&&current.positions[i]<start+r.offset+r.count){
                if(r.site>=6){auto &seen=current.final_seen[current.group[i]*2+(r.site==10)];check(!seen,"Evaluation repeated final row");seen=1;continue;}
                for(uint32_t site=r.site;site<(r.site==0?4:r.site+1);++site){
                    auto &seen=current.seen[(size_t(layer)*current.positions.size()+i)*6+site];
                    check(!seen,"Writer capture repeated a selected layer/site");seen=1;
                }
            }
            for(uint32_t group=0;group<current.groups;++group)for(uint32_t c=0;c<r.width;++c){
                const auto v=values[size_t(group)*r.width+c];check(std::isfinite(v),"Nonfinite writer capture value");
                if(r.site==6){current.final_residual[size_t(group)*r.width+c]+=v;continue;}
                if(r.site==10){current.final_pre[size_t(group)*r.width+c]+=v;continue;}
                const auto site=r.site==0?c/5120:r.site;
                current.output[((size_t(group)*40+layer)*6+site)*5120+c%5120]+=v;
            }
        }
    }
};
}
