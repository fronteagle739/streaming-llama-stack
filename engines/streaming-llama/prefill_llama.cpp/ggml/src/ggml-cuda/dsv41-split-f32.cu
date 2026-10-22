// Reuses NVIDIA CUTLASS v3.9.2 OpMultiplyAddFastF32, example27 and
// gemm/warp/mma_tensor_op_fast_f32.h. Dependency revision/license recorded in
// DS4PERF_SPLIT_F32.md. Three TF32 products approximate F32; gate numerically.
#include "dsv41-split-f32.cuh"
#if defined(GGML_CUDA_DSV41_SPLIT_F32)
#include <cutlass/cutlass.h>
#include <cutlass/platform/platform.h>
#include <type_traits>
#if defined(_MSC_VER)
// Pinned CUTLASS3.9.2's Windows trait set omits this variable template used
// by exmy_base.h. Keep the compatibility alias here; upstream stays pristine.
namespace cutlass { namespace platform {
template<class T> constexpr bool is_unsigned_v = std::is_unsigned<T>::value;
}}
#endif
#include <cutlass/gemm/device/gemm_batched.h>

template<int M, int N, int WM, int WN, typename TransposedValueLayout=cutlass::layout::ColumnMajor>
static int launch(int m, int n, int k,
    const float * a, int lda, int64_t batch_a,
    const float * b, int ldb, int64_t batch_b,
    float * c, int ldc, int64_t batch_c, int batches, cudaStream_t stream) {
    using Gemm = cutlass::gemm::device::GemmBatched<
        float, cutlass::layout::RowMajor, float, TransposedValueLayout,
        float, cutlass::layout::RowMajor, float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<M,N,32>,
        cutlass::gemm::GemmShape<WM,WN,32>,
        cutlass::gemm::GemmShape<16,8,8>,
        cutlass::epilogue::thread::LinearCombination<float,4,float,float>,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        3,4,4,cutlass::arch::OpMultiplyAddFastF32>;
    // CUTLASS3.9.2's column-major GemmBatched specialization omits Operator_
    // when forwarding to its underlying row-major operator (lines542-560).
    // Express C^T=B^T*A^T directly so FastF32 cannot silently become 1xTF32.
    static_assert(std::is_same<typename Gemm::Operator,
        cutlass::arch::OpMultiplyAddFastF32>::value, "Three-term operator required");
    typename Gemm::Arguments args({n,m,k}, {b,ldb},batch_b,{a,lda},batch_a,
        {c,ldc},batch_c,{c,ldc},batch_c,{1.0f,0.0f},batches);
    Gemm op;
    auto status=op.can_implement(args);
    if(status!=cutlass::Status::kSuccess)return int(status);
    // No split-K or workspace allocation is allowed in this bounded path.
    if(Gemm::get_workspace_size(args)!=0)return -2;
    return int(op(args,nullptr,stream));
}

int ggml_cuda_dsv41_split_f32(int tile, int m, int n, int k,
    const float * a, int lda, int64_t batch_a,
    const float * b, int ldb, int64_t batch_b,
    float * c, int ldc, int64_t batch_c, int batches, cudaStream_t stream) {
    if(m<=0||n<=0||k<=0||batches<=0||!a||!b||!c||lda<k||ldb<k||ldc<m||
       m%4||k%4||lda%4||ldb%4||ldc%4||batch_a<0||batch_b<0||batch_c<0||
       (batches>1&&((batch_a&&batch_a<int64_t(lda)*m)||
                    (batch_b&&batch_b<int64_t(ldb)*n)||batch_c<int64_t(ldc)*n))||
       ((uintptr_t(a)|uintptr_t(b)|uintptr_t(c))&15))return -1;
    if(tile==0)return launch<128,64,64,32>(m,n,k,a,lda,batch_a,b,ldb,batch_b,c,ldc,batch_c,batches,stream);
    if(tile==1)return launch<64,64,32,32>(m,n,k,a,lda,batch_a,b,ldb,batch_b,c,ldc,batch_c,batches,stream);
    if(tile==2)return launch<32,64,16,32>(m,n,k,a,lda,batch_a,b,ldb,batch_b,c,ldc,batch_c,batches,stream);
    return -1;
}

int ggml_cuda_dsv41_split_f32_pv(int m, int n, int k,
    const float * a, int lda, int64_t batch_a,
    const float * b, int ldb, int64_t batch_b,
    float * c, int ldc, int64_t batch_c, int batches, cudaStream_t stream) {
    if(m!=512||n!=64||k<4||k>640||k%4||batches<1||batches>4096||
       !a||!b||!c||lda!=m||ldb!=k||ldc!=m||batch_a!=int64_t(m)*k||
       batch_b!=int64_t(k)*n||batch_c!=int64_t(m)*n||
       ((uintptr_t(a)|uintptr_t(b)|uintptr_t(c))&15))return -1;
    return launch<64,64,32,32,cutlass::layout::RowMajor>(m,n,k,a,lda,batch_a,b,ldb,batch_b,c,ldc,batch_c,batches,stream);
}
#else
int ggml_cuda_dsv41_split_f32(int, int, int, int, const float *, int, int64_t,
    const float *, int, int64_t, float *, int, int64_t, int, cudaStream_t) {
    return -3; // Explicitly unavailable unless this optional dependency is built.
}
int ggml_cuda_dsv41_split_f32_pv(int,int,int,const float*,int,int64_t,
    const float*,int,int64_t,float*,int,int64_t,int,cudaStream_t){return -3;}
#endif
