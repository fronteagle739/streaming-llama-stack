// Metadata-only tests: no backend, device or tensor storage allocation.
#pragma once
#include "ggml-qfn-attention.h"

template<class Expect>
static void qfn_attention_node_contract(ggml_context * ctx,Expect expect) {
    for(int nt:{1,4,17}) for(bool gated:{false,true}) {
        constexpr int D=256,H=24,Hkv=2,N=257,cap=67;
        auto * qp=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,H+1,nt+1);
        auto * kp=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,D+8,Hkv+1,N+1);
        auto * vp=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,D+16,Hkv+2,N+2);
        auto * ip=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,cap+3,nt);
        auto * q=ggml_view_3d(ctx,qp,D,H,nt,qp->nb[1],qp->nb[2],0);
        auto * k=ggml_view_3d(ctx,kp,D,Hkv,N,kp->nb[1],kp->nb[2],0);
        auto * v=ggml_view_3d(ctx,vp,D,Hkv,N,vp->nb[1],vp->nb[2],0);
        auto * ids=ggml_view_2d(ctx,ip,cap,nt,ip->nb[1],0);
        auto * pos=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,nt);
        auto * gp=gated?ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,H+2,nt+2):nullptr;
        auto * gate=gp?ggml_view_3d(ctx,gp,D,H,nt,gp->nb[1],gp->nb[2],0):nullptr;
        const uint64_t need=uint64_t(nt)*4*((uint64_t(cap)+63)/64*H*258+64);
        auto * scratch=ggml_new_tensor_1d(ctx,GGML_TYPE_I8,int64_t(need));
        auto * t=ggml_qfn_attention(ctx,q,gate,k,v,ids,pos,0.0625f,nt<=4?scratch:nullptr);
        expect(ggml_qfn_attention_valid(t),"attention padded metadata rejected");
        expect(t->src[0]==q && t->src[1]==gate && t->src[2]==k && t->src[3]==v &&
               t->src[4]==ids && t->src[5]==pos && t->src[6]==(nt<=4?scratch:nullptr),
               "attention producer/scratch dependencies lost");
        auto reject=[&](ggml_tensor * tensor,auto mutate,const char * reason) {
            const auto saved=*tensor;mutate(*tensor);
            const bool accepted=ggml_qfn_attention_valid(t);*tensor=saved;
            expect(!accepted,reason);
        };
        reject(t,[](auto & x){x.op_params[1]=0;},"native tag accepted as Basalt attention");
        reject(t,[](auto & x){x.op_params[1]=2;},"future attention tag accepted");
        reject(t,[](auto & x){x.ne[3]=2;},"unsupported output stream axis accepted");
        reject(t,[](auto & x){x.nb[1]=512;},"overlapping output heads accepted");
        reject(t,[](auto & x){x.nb[2]=1024;},"overlapping output tokens accepted");
        reject(t,[](auto & x){float v=std::numeric_limits<float>::infinity();std::memcpy(x.op_params,&v,4);},
               "nonfinite attention scale accepted");
        reject(q,[](auto & x){x.nb[0]=8;},"strided query elements accepted");
        reject(q,[](auto & x){x.nb[1]+=4;},"misaligned query head stride accepted");
        reject(q,[](auto & x){x.nb[1]=SIZE_MAX-15;},"overflowing query byte span accepted");
        reject(q,[](auto & x){x.data=reinterpret_cast<void *>(UINTPTR_MAX-15);},"overflowing query pointer accepted");
        reject(k,[](auto & x){x.type=GGML_TYPE_F32;},"unsupported KV precision accepted");
        reject(k,[](auto & x){x.ne[1]=3;},"incompatible grouped query geometry accepted");
        reject(k,[](auto & x){x.nb[2]+=2;},"misaligned key cell stride accepted");
        reject(v,[](auto & x){x.ne[2]--;},"mismatched independent value shape accepted");
        reject(v,[](auto & x){x.nb[1]+=2;},"misaligned independent value stride accepted");
        reject(ids,[](auto & x){x.nb[1]+=2;},"misaligned selection stride accepted");
        reject(ids,[](auto & x){x.ne[0]=int64_t(INT32_MAX)+1;},"oversized selection capacity accepted");
        if(nt>1) reject(pos,[](auto & x){x.nb[0]=8;},"noncontiguous positions accepted");
        reject(pos,[](auto & x){x.ne[0]++;},"incompatible position shape accepted");
        if(gate) {
            reject(gate,[](auto & x){x.type=GGML_TYPE_F16;},"unsupported gate precision accepted");
            reject(gate,[](auto & x){x.ne[1]--;},"incompatible optional gate shape accepted");
            reject(gate,[](auto & x){x.nb[1]+=2;},"misaligned optional gate stride accepted");
        }
        if(nt<=4) {
            reject(t,[](auto & x){x.src[6]=nullptr;},"required decode scratch omitted");
            reject(scratch,[&](auto & x){x.ne[0]=int64_t(need)-1;},"undersized decode scratch accepted");
            reject(scratch,[](auto & x){x.type=GGML_TYPE_F32;},"wrong scratch element type accepted");
            reject(scratch,[](auto & x){x.data=reinterpret_cast<void *>(256+1);},"misaligned decode scratch pointer accepted");
        } else {
            reject(t,[&](auto & x){x.src[6]=scratch;},"prompt scratch field silently ignored");
        }
        const auto saved_q=*q; q->nb[1]=q->nb[2]=0;
        expect(ggml_qfn_attention_valid(t),"supported query broadcasting rejected");*q=saved_q;
        const auto saved_ids=*ids;ids->ne[0]=0;
        expect(ggml_qfn_attention_valid(t),"empty selected list rejected");*ids=saved_ids;
        const auto saved_k=*k,saved_v=*v;k->ne[2]=v->ne[2]=0;
        expect(ggml_qfn_attention_valid(t),"empty KV pool rejected");*k=saved_k;*v=saved_v;
        const auto saved=*t;float zero=0;std::memcpy(t->op_params,&zero,4);
        expect(ggml_qfn_attention_valid(t),"supported finite zero scale rejected");*t=saved;
    }
}
