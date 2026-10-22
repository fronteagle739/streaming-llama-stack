#ifndef DS4_LORA_H
#define DS4_LORA_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    float *a, *b;
    uint32_t rank;
    float alpha;
} ds4_output_lora;
/* Only the dense output projection is supported. Base weights stay read-only. */
int ds4_output_lora_read(const char *path, const char *base_sha256,
                        uint32_t hidden, uint32_t vocab,
                        ds4_output_lora *out, char *error, size_t error_cap);
void ds4_output_lora_free(ds4_output_lora *adapter);
#ifdef __cplusplus
}
#endif
#endif
