#include "llama-dsv41-profile-stage.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
static void check(bool ok){if(!ok)throw std::runtime_error("Diagnostic graph naming contract failed");}
int main(){try{
    auto*c=ggml_init({1024*1024,nullptr,true});check(c);
    auto*a=ggml_new_tensor_2d(c,GGML_TYPE_F32,16,3);auto*b=ggml_new_tensor_2d(c,GGML_TYPE_F32,16,3);
    ggml_tensor*inner,*outer,*semantic,*off;
    {llama_dsv41_profile_stage parent(c,"hc",true);
        {llama_dsv41_profile_stage child(c,"norm",true);inner=ggml_rms_norm(c,a,1e-6f);}
        outer=ggml_add(c,inner,b);semantic=ggml_mul(c,a,b);ggml_set_name(semantic,"dsv41_dispatch_boundary");}
    check(std::strncmp(inner->name,"diag_norm_",10)==0);check(std::strncmp(outer->name,"diag_hc_",8)==0);
    check(std::strcmp(semantic->name,"dsv41_dispatch_boundary")==0);
    check(inner->op==GGML_OP_RMS_NORM&&inner->src[0]==a&&outer->op==GGML_OP_ADD&&outer->src[0]==inner&&outer->src[1]==b);
    {llama_dsv41_profile_stage disabled(c,"off",false);off=ggml_add(c,a,b);}
    check(std::strncmp(off->name,"diag_",5)!=0);check(a->name[0]==0&&b->name[0]==0);
    ggml_free(c);std::cout<<"PASS nested/disabled naming, dispatch-name preservation and unchanged operation/operand identities\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
