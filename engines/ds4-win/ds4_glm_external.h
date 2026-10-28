/* Experimental in-process GLM-5.3 decode ABI. Callers own all weights/state.
 * Layouts are native ds4 CUDA layouts, not serialized llama state blobs.
 * Conv: [Q/K/V][history][head*channel] F32; recurrent: [head][value][key]
 * F32; latent: [token][kv_lora] F16; pool: [pool][index_dim] F16;
 * tails: [token % 4][index_dim] F32. No BF16/F16 reinterpretation is allowed.
 * This ABI does not itself provide ULTRA prefill, state conversion or serving.
 */
#ifndef DS4_GLM_EXTERNAL_H
#define DS4_GLM_EXTERNAL_H
#include <stdint.h>
#include <stddef.h>
#define DS4GLM_ABI 1u
typedef struct {
    void *data;
    uint64_t bytes;
} ds4glm_span;
typedef struct {
    ds4glm_span conv, recurrent, latent, pooled_key, tail_key, tail_gate;
} ds4glm_layer_state;
typedef struct {
    uint32_t abi, layers, context, position;
    const ds4glm_layer_state *layer;
} ds4glm_state;
#if defined(_WIN32) && defined(DS4_EXTERNAL_DECODE)
#define DS4GLM_API __declspec(dllexport)
#else
#define DS4GLM_API
#endif
/* Exported by DS4_EXTERNAL_DECODE builds only. Resolve these from the decode
 * library in the producer; normal native builds do not expose the adapter. */
#ifdef __cplusplus
extern "C" {
#endif
DS4GLM_API int ds4glm_open(const char *path, uint32_t context, uint64_t scratch_limit,
    char *error, size_t cap);
DS4GLM_API int ds4glm_register_weight(uint64_t offset, uint64_t bytes, void *ptr);
DS4GLM_API int ds4glm_replace_weight(uint64_t offset, uint64_t bytes, void *ptr);
DS4GLM_API uint64_t ds4glm_drop_weights(void);
DS4GLM_API int ds4glm_bind_state(const ds4glm_state *state, char *error, size_t cap);
DS4GLM_API int ds4glm_decode(int32_t token, uint32_t position, float *logits);
DS4GLM_API void ds4glm_close(void);
#ifdef __cplusplus
}
#endif
#endif
