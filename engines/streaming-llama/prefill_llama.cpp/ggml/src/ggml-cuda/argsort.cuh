#include "common.cuh"

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#ifdef GGML_CUDA_USE_CUB
void argsort_f32_i32_cuda_cub(ggml_cuda_pool & pool,
                              const float *    x,
                              int *            dst,
                              const int        ncols,
                              const int        nrows,
                              ggml_sort_order  order,
                              cudaStream_t     stream);
#endif  // GGML_CUDA_USE_CUB
void argsort_f32_i32_cuda_bitonic(const float *   x,
                                  int *           dst,
                                  const int       ncols,
                                  const int       nrows,
                                  ggml_sort_order order,
                                  cudaStream_t    stream);
void top_k_f32_i32_cuda_bitonic(const float * x,
                                int *         dst,
                                const int     ncols,
                                const int     nrows,
                                const int     k,
                                cudaStream_t  stream);
void top_k_f32_i32_cuda_bitonic_tiles(const float * x,
                                      int *         dst,
                                      const int     ncols,
                                      const int     nrows,
                                      const int     tile_size,
                                      const int     k,
                                      cudaStream_t  stream);
void top_k_f32_i32_cuda_bitonic_tiles_guarded(const float *x,int *dst,int ncols,int nrows,
                                             int tile_size,int k,cudaStream_t stream,bool reject_nan);
void top_k_f32_i32_cuda_bitonic_candidates(const float * x,
                                           const int *   candidates,
                                           int *         dst,
                                           const int     ncols,
                                           const int     nrows,
                                           const int     width,
                                           const int     tile_size,
                                           const int     k,
                                           cudaStream_t  stream);

void cuda_bailingmoev2_experts(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * topk_view);

void cuda_glm45moe_experts(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * topk_view);

void cuda_openai_experts(ggml_backend_cuda_context & ctx, ggml_tensor * topk, ggml_tensor * softmax);
