// Guarded real-model bring-up for GLM's existing streaming graph. This is not
// the qualified ULTRA/ds4 producer. Run only inside the stack memory/GPU guard.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc,char **argv) {
    if(argc<5 || argc>6) {
        fprintf(stderr,"usage: %s MODEL PROMPT_FILE CONTEXT DECODE_TOKENS [FOLLOWUP_SUFFIX_FILE]\n",argv[0]);return 2;
    }
    const char *wf=getenv("LLAMA_PREFILL_WF");
    if(!wf || std::string(wf)!="1") {
        fprintf(stderr,"GLM_STREAM_REFUSED: LLAMA_PREFILL_WF=1 required; no resident load fallback\n");return 2;
    }
    std::ifstream file(argv[2],std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(file)),{});
    if(!file || text.empty() || text.size()>4*1024*1024) return 2;
    const int context=atoi(argv[3]),decode=atoi(argv[4]);
    if(context<128 || context>1048576 || decode<0 || decode>128)return 2;
    llama_backend_init();
    auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.no_host=true;mp.use_mmap=false;
    auto *model=llama_model_load_from_file(argv[1],mp);
    if(!model)return 3;
    const auto *vocab=llama_model_get_vocab(model);
    int count=-llama_tokenize(vocab,text.data(),(int)text.size(),nullptr,0,true,true);
    if(count<=0 || count+decode>context)return 4;
    std::vector<llama_token> tokens(count);
    count=llama_tokenize(vocab,text.data(),(int)text.size(),tokens.data(),count,true,true);
    if(count<=0)return 4;
    std::vector<llama_token> followup;
    if(argc==6) {
        std::ifstream next(argv[5],std::ios::binary);
        std::string suffix((std::istreambuf_iterator<char>(next)),{});
        if(!next || suffix.empty() || suffix.size()>4*1024*1024)return 4;
        int n=-llama_tokenize(vocab,suffix.data(),(int)suffix.size(),nullptr,0,false,true);
        if(n<=0)return 4;followup.resize(n);
        if(llama_tokenize(vocab,suffix.data(),(int)suffix.size(),followup.data(),n,false,true)!=n)return 4;
    }
    auto cp=llama_context_default_params();
    cp.n_ctx=context;cp.n_batch=std::max({count,(int)followup.size(),256});cp.n_ubatch=256;
    cp.n_threads=2;cp.n_threads_batch=2;cp.n_seq_max=1;
    cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.prefill_streaming=true;cp.prefill_only=false;
    cp.prefill_layer_buffers=1;cp.prefill_overlap=false;
    cp.prefill_min_stream_batch_tokens=1;
    cp.prefill_decode_mode=LLAMA_PREFILL_DECODE_MODE_STREAM_LOCAL;
    auto *ctx=llama_init_from_model(model,cp);if(!ctx)return 5;
    auto batch=llama_batch_init(cp.n_batch,0,1);
    int position_base=0,frontier=0;
    // Process the uniform bulk before the exact tail. Padding a recurrent
    // microbatch advances KDA state through fake tokens and changes appends.
    auto submit=[&](int first,int length) {
        for(int i=0;i<length;++i) {
            batch.token[i]=tokens[first+i];batch.pos[i]=position_base+first+i;
            batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;
            batch.logits[i]=i==length-1;
        }
        batch.n_tokens=length;return llama_decode(ctx,batch);
    };
    auto start=std::chrono::steady_clock::now();
    const int bulk=count/cp.n_ubatch*cp.n_ubatch;
    int code=bulk ? submit(0,bulk) : 0;
    if(!code && bulk<count)code=submit(bulk,count-bulk);
    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    fprintf(stderr,"GLM_STREAM_PREFILL tokens=%d seconds=%.6f tok_s=%.6f code=%d\n",count,elapsed,count/elapsed,code);
    if(code)return 6;
    frontier=count;
    const int nv=llama_vocab_n_tokens(vocab);
    auto generate=[&](int turn,bool commit_eog) {
    for(int i=0;i<decode;++i) {
        const float *logits=llama_get_logits_ith(ctx,-1);
        int token=(int)(std::max_element(logits,logits+nv)-logits);
        char piece[1024];int bytes=llama_token_to_piece(vocab,token,piece,sizeof(piece),0,true);
        if(bytes>0)fwrite(piece,1,bytes,stdout);fflush(stdout);
        fprintf(stderr,"GLM_STREAM_TOKEN step=%d token=%d eog=%d turn=%d\n",i,token,(int)llama_vocab_is_eog(vocab,token),turn);
        bool end=llama_vocab_is_eog(vocab,token);
        if(end && !commit_eog)return 1;
        batch.n_tokens=1;batch.token[0]=token;batch.pos[0]=frontier;batch.logits[0]=1;
        if(llama_decode(ctx,batch))return -1;
        ++frontier;
        if(end)return 1;
    }
    return 0;
    };
    int generated=generate(0,!followup.empty());
    if(generated<0)return 7;
    if(!followup.empty()) {
        // The suffix follows the committed model end/next-role token. The
        // caller supplies its chat-template continuation, without a new BOS.
        if(generated!=1 || frontier+(int)followup.size()+decode>context)return 8;
        position_base=frontier;tokens=followup;count=(int)tokens.size();
        start=std::chrono::steady_clock::now();
        const int next_bulk=count/cp.n_ubatch*cp.n_ubatch;
        code=next_bulk ? submit(0,next_bulk) : 0;
        if(!code && next_bulk<count)code=submit(next_bulk,count-next_bulk);
        elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        fprintf(stderr,"GLM_STREAM_APPEND reused=%d tokens=%d seconds=%.6f code=%d\n",position_base,count,elapsed,code);
        if(code)return 9;frontier+=count;
        if(generate(1,false)<0)return 10;
    }
    llama_batch_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();
    return 0;
}
