#pragma once
#include <stdint.h>
// Experimental fixture entry only: all pointers device-owned and contiguous.
// q/out[512,heads,queries], kv[512,rows], ids/mask[topk,queries], sinks[heads].
// Caller admits RAM/VRAM and validates finite half-representable Q/K, finite
// sinks and valid IDs; masked KV cells may be nonfinite and must not be loaded.
// Uses caller stream, never allocates or changes canonical inputs. No fallback.
#ifdef __cplusplus
extern "C" {
#endif
bool ggml_dsv41_cuda_sparse_f32_replay(const float*,const float*,const int32_t*,const float*,const float*,float*,int,int,int,int,void*);
uint64_t ggml_dsv41_cuda_sparse_f32_replay_queued();
#ifdef __cplusplus
}
#endif
