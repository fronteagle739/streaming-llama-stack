# SPDX-License-Identifier: MIT
# Copyright (c) 2023 DeepSeek
# Adapted from pinned upstream/kernel.py:sparse_attn_kernel. See upstream/LICENSE.
# Keep the official BF16 MMA/online-softmax arithmetic. Stage native queries/KV
# once; one grid covers all queries and head groups without128-row relaunches.
import tilelang
import tilelang.language as T

@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED:True,
                          tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER:True})
def native_sparse_attn_kernel():
    m=T.symbolic('m');n=T.symbolic('n')
    @T.prim_func
    def native_sparse_attn_kernel_(
        q:T.Tensor[(m,64,512),'bfloat16'],
        kv:T.Tensor[(n,512),'bfloat16'],
        out:T.Tensor[(m,64,512),'float32'],
        sink:T.Tensor[(64,),'float32'],
        ids:T.Tensor[(m,640),'int32']):
        with T.Kernel(m,4,threads=256) as (row,group):
            qs=T.alloc_shared((16,512),'bfloat16')
            ks=T.alloc_shared((64,512),'bfloat16')
            os=T.alloc_shared((16,512),'bfloat16')
            ps=T.alloc_shared((16,64),'bfloat16')
            selected=T.alloc_fragment(64,'int32')
            scores=T.alloc_fragment((16,64),'float32')
            accum=T.alloc_fragment((16,512),'float32')
            maximum=T.alloc_fragment(16,'float32')
            previous=T.alloc_fragment(16,'float32')
            correction=T.alloc_fragment(16,'float32')
            block_sum=T.alloc_fragment(16,'float32')
            total=T.alloc_fragment(16,'float32')
            T.clear(accum);T.clear(total);T.fill(maximum,-1e30)
            for h,d in T.Parallel(16,512):
                qs[h,d]=q[row,group*16+h,d]
            for tile in T.Pipelined(10,num_stages=1):
                for j in T.Parallel(64):selected[j]=ids[row,tile*64+j]
                for j,d in T.Parallel(64,512):
                    ks[j,d]=T.if_then_else(selected[j]>=0,kv[selected[j],d],0)
                for h,j in T.Parallel(16,64):scores[h,j]=T.if_then_else(selected[j]>=0,0,-T.infinity('float32'))
                T.gemm(qs,ks,scores,transpose_B=True,policy=T.GemmWarpPolicy.FullRow)
                for h,j in T.Parallel(16,64):scores[h,j]*=512**-.5
                T.copy(maximum,previous)
                T.reduce_max(scores,maximum,dim=1,clear=False)
                for h in T.Parallel(16):correction[h]=T.exp(previous[h]-maximum[h])
                for h,j in T.Parallel(16,64):scores[h,j]=T.exp(scores[h,j]-maximum[h])
                T.reduce_sum(scores,block_sum,dim=1)
                for h in T.Parallel(16):total[h]=total[h]*correction[h]+block_sum[h]
                T.copy(scores,ps)
                for h,d in T.Parallel(16,512):accum[h,d]*=correction[h]
                T.gemm(ps,ks,accum,policy=T.GemmWarpPolicy.FullRow)
            for h in T.Parallel(16):total[h]+=T.exp(sink[group*16+h]-maximum[h])
            for h,d in T.Parallel(16,512):accum[h,d]/=total[h]
            T.copy(accum,os)
            for h,d in T.Parallel(16,512):out[row,group*16+h,d]=os[h,d]
    return native_sparse_attn_kernel_
