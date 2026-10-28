#pragma once
#include "ggml.h"
#include <cstdlib>
#include <cstring>

// Diagnostic names only. Preserve every semantic dispatch/boundary name and
// every existing operand, operation, layout and precision flag.
class llama_dsv41_profile_stage {
    ggml_context *context=nullptr;
    ggml_tensor *last=nullptr;
    const char *stage;
    static bool enabled(){
        static const bool value=[](){const char *s=std::getenv("LLAMA_DSV41_PROFILE_STAGES");return s&&s[0]=='1';}();
        return value;
    }
public:
    llama_dsv41_profile_stage(ggml_context*c,const char*s):llama_dsv41_profile_stage(c,s,enabled()){}
    llama_dsv41_profile_stage(ggml_context*c,const char*s,bool on):context(on?c:nullptr),stage(s){
        if(context)for(auto*t=ggml_get_first_tensor(context);t;t=ggml_get_next_tensor(context,t))last=t;
    }
    ~llama_dsv41_profile_stage(){finish();}
    void finish(){
        if(!context)return;
        unsigned serial=0;
        for(auto*t=last?ggml_get_next_tensor(context,last):ggml_get_first_tensor(context);t;t=ggml_get_next_tensor(context,t)){
            const char*n=t->name;
            const bool anonymous=!n[0]||n[0]==' '||std::strncmp(n,"node_",5)==0||std::strncmp(n,"leaf_",5)==0;
            if(t->op!=GGML_OP_NONE&&anonymous)ggml_format_name(t,"diag_%s_%u",stage,serial++);
        }
        context=nullptr;
    }
    llama_dsv41_profile_stage(const llama_dsv41_profile_stage&)=delete;
    llama_dsv41_profile_stage&operator=(const llama_dsv41_profile_stage&)=delete;
};
