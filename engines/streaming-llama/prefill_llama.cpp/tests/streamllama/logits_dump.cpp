// Dump the logits of a deterministic prompt (+ a few single-token decode steps) for fork-vs-upstream comparison.
// Uses public llama/ggml APIs so the same source can build against upstream.
//
// usage: logits-dump model.gguf out.bin n_prompt n_ubatch [n_decode=8]
//        [flash_attn=0] [seed=1234] [gpu_layers=0] [restore=0]
//        [trace_dir=-] [force_f32=0] [streaming=0] [last_only=0]
// out.bin: int32 n_rows, int32 n_vocab, then n_rows*n_vocab float32 (prompt rows, then one row per decode step)
// Tracing reads synchronized tensors via the scheduler callback, adds no nodes,
// and pins no buffers. It splits execution and is diagnostic only. Confirm fixes
// with trace_dir=- and force_f32=0 (no callback at all).

#include "llama.h"
#include "ggml-backend.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <cstring>
struct probe { std::string dir; FILE * manifest=nullptr; llama_context * context=nullptr; int phase=0, node=0, strict=0, strict_count=0; };
static bool observe(ggml_tensor * t, bool ask, void * data) {
    auto & p=*static_cast<probe *>(data);
    if (ask) {
        if (p.strict && (t->op==GGML_OP_MUL_MAT || t->op==GGML_OP_MUL_MAT_ID)) {
            t->op_params[0]=GGML_PREC_F32;
            if (++p.strict_count<=8) fprintf(stderr,"PROBE F32 engaged name=%s op=%s\n",t->name,ggml_op_name(t->op));
        }
        return p.manifest && t->type==GGML_TYPE_F32 && t->name[0];
    }
    // Streaming deliberately skips scheduler callback synchronization. A
    // device read on the default stream alone does not order the compute
    // stream, so establish completion before observing any tensor.
    if(p.context)llama_synchronize(p.context);
    char stem[64]; snprintf(stem,sizeof(stem),"%d-%d",p.phase,p.node++);
    fprintf(p.manifest,"%d\t%s\t%s\t%s",p.phase,t->name,ggml_op_name(t->op),stem);
    for(int i=0;i<4;++i) fprintf(p.manifest,"\t%lld",(long long)t->ne[i]);
    fprintf(p.manifest,"\n"); fflush(p.manifest);
    std::vector<unsigned char> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t,bytes.data(),0,bytes.size());
    FILE * f=fopen((p.dir+"/"+stem+".bin").c_str(),"wb"); if(!f) std::abort();
    for(int64_t i3=0;i3<t->ne[3];++i3) for(int64_t i2=0;i2<t->ne[2];++i2)
    for(int64_t i1=0;i1<t->ne[1];++i1) for(int64_t i0=0;i0<t->ne[0];++i0) {
        float value; memcpy(&value,bytes.data()+i0*t->nb[0]+i1*t->nb[1]+i2*t->nb[2]+i3*t->nb[3],4);
        fwrite(&value,4,1,f);
    }
    fclose(f); return true;
}


int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s model.gguf out.bin n_prompt n_ubatch [n_decode] [flash_attn] [seed] [gpu_layers] [restore] [trace_dir=-] [force_f32=0] [streaming=0] [last_only=0]\n", argv[0]);
        return 1;
    }
    const char * path_model = argv[1];
    const char * path_out   = argv[2];
    const int n_prompt = atoi(argv[3]);
    const int n_ubatch = atoi(argv[4]);
    const int n_decode = argc > 5 ? atoi(argv[5]) : 8;
    const int fa       = argc > 6 ? atoi(argv[6]) : 0;
    const uint32_t seed = argc > 7 ? (uint32_t) atoi(argv[7]) : 1234u;
    const bool last_only=argc>13 && atoi(argv[13]);
    const bool streaming=argc>12 && atoi(argv[12]);
    if(streaming && (!getenv("LLAMA_PREFILL_WF") || strcmp(getenv("LLAMA_PREFILL_WF"),"1"))) {
        fprintf(stderr,"Streaming comparison requires LLAMA_PREFILL_WF=1 under the GPU/memory guardian\n");return 2;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = argc > 8 ? atoi(argv[8]) : 0;
    llama_model * model = llama_model_load_from_file(path_model, mparams);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    probe p;
    if(argc>10 && strcmp(argv[10],"-")!=0) {
        p.dir=argv[10]; p.manifest=fopen((p.dir+"/tensors.tsv").c_str(),"w");
        if(!p.manifest) return 4;
    }
    p.strict=argc>11 ? atoi(argv[11]):0;
    fprintf(stderr,"PROBE graphs_disabled=%s strict=%d tracing=%d\n",getenv("GGML_CUDA_DISABLE_GRAPHS")?getenv("GGML_CUDA_DISABLE_GRAPHS"):"unset",p.strict,p.manifest!=nullptr);
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 512;
    cparams.n_batch         = n_prompt;
    cparams.n_ubatch        = n_ubatch;
    cparams.n_seq_max       = 1;
    cparams.n_threads       = 2;
    cparams.n_threads_batch = 2;
    cparams.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if(p.strict || p.manifest) { cparams.cb_eval=observe; cparams.cb_eval_user_data=&p; }
    cparams.no_perf         = true;
    cparams.type_k = GGML_TYPE_F32;
    cparams.type_v = GGML_TYPE_F32;
    if(streaming) {
        cparams.prefill_streaming=true;
        cparams.prefill_decode_mode=LLAMA_PREFILL_DECODE_MODE_STREAM_LOCAL;
        cparams.prefill_min_stream_batch_tokens=1;
        fprintf(stderr,"GLM_STREAM_LOGITS requested=1 prompt=%d microbatch=%d\n",n_prompt,n_ubatch);
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }
    p.context=ctx;

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // deterministic tokens (LCG)
    uint32_t s = seed;
    auto next_tok = [&]() {
        s = s * 1664525u + 1013904223u;
        return (llama_token) ((s >> 8) % (uint32_t) n_vocab);
    };

    std::vector<float> out;

    llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; ++i) {
        batch.token[i]     = next_tok();
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = !last_only || i==n_prompt-1;
    }
    batch.n_tokens = n_prompt;

    auto prompt_part = [&](llama_batch part, bool collect=true) {
        if (llama_decode(ctx, part) != 0) return false;
        if(last_only) {
            if(collect) {const float *l=llama_get_logits_ith(ctx,-1);out.insert(out.end(),l,l+n_vocab);}
            return true;
        }
        for (int i=0;i<part.n_tokens;++i) {
            const float *l=llama_get_logits_ith(ctx,i);
            out.insert(out.end(),l,l+n_vocab);
        }
        return true;
    };
    // A recurrent state must never absorb padded tail tokens. Exercise the
    // layer-major prefix and its exact-sized ragged continuation separately.
    const int bulk=streaming ? n_prompt/n_ubatch*n_ubatch : n_prompt;
    bool prompt_ok=true;
    if(streaming && bulk>0 && bulk<n_prompt) {
        auto prefix=batch;prefix.n_tokens=bulk;
        prompt_ok=prompt_part(prefix,!last_only);
        auto tail=batch;tail.n_tokens=n_prompt-bulk;
        tail.token+=bulk;tail.pos+=bulk;tail.n_seq_id+=bulk;tail.seq_id+=bulk;tail.logits+=bulk;
        if(prompt_ok)prompt_ok=prompt_part(tail);
    } else prompt_ok=prompt_part(batch);
    if(!prompt_ok){fprintf(stderr,"prompt decode failed\n");return 1;}

    // Restore into cleared caches so missing indexer/recurrent owners cannot pass
    // merely because their old bytes survived in the same context.
    if (argc > 9 && atoi(argv[9])) {
        std::vector<uint8_t> state(llama_state_get_size(ctx));
        const size_t saved = llama_state_get_data(ctx, state.data(), state.size());
        if (!saved) return 2;
        llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_state_set_data(ctx, state.data(), saved) != saved) return 3;
    }

    p.phase=1;
    for (int j = 0; j < n_decode; ++j) {
        batch.token[0]     = next_tok();
        batch.pos[0]       = n_prompt + j;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;
        batch.n_tokens     = 1;
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode step %d failed\n", j);
            return 1;
        }
        const float * l = llama_get_logits_ith(ctx, 0);
        out.insert(out.end(), l, l + n_vocab);
    }

    FILE * f = fopen(path_out, "wb");
    const int32_t hdr[2] = { (last_only ? 1 : n_prompt) + n_decode, n_vocab };
    fwrite(hdr, sizeof(int32_t), 2, f);
    fwrite(out.data(), sizeof(float), out.size(), f);
    fclose(f);

    fprintf(stderr, "wrote %d rows x %d logits to %s\n", hdr[0], hdr[1], path_out);

    if(p.manifest) fclose(p.manifest);
    fprintf(stderr,"PROBE strict_nodes=%d\n",p.strict_count);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
