#include "ggml-alloc.h"
#include "ggml-dsv41-bf16-inplace.h"
#include "ggml-dsv41-rope-inplace.h"
#include "ggml-dsv41-masked-half-inplace.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-impl.h"
#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX_FREE_BLOCKS 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#define AT_PRINTF(...)


static bool ggml_is_view(const struct ggml_tensor * t) {
    return t->view_src != NULL;
}

// ops that return true for this function must not use restrict pointers for their backend implementations
// 2026-05-28: gallocr buffer-reuse toggles. The multi-ubatch LAYER-MAJOR prefill
// executor re-runs each layer's node-range slice once PER UBATCH against the buffer
// layout that gallocr froze for a SINGLE in-order full-graph pass. A buffer freed or
// in-place-reused mid-layer then gets clobbered before a still-needed consumer in the
// repeated L1 (first-MoE, high-intermediate-pressure) slice reads it -> deterministic
// L1 corruption (single-ub, which runs one pass, is unaffected). These toggles make the
// frozen layout valid for the sliced/repeated execution.
//   GGML_ALLOC_NO_INPLACE     - disable in-place parent reuse (cheap; ~0 extra VRAM)
//   GGML_ALLOC_NO_FREE_REUSE  - disable free-then-reassign (full no-reuse; high VRAM)
// Read once; default OFF so production (env unset) is byte-identical & full-speed.
static bool ggml_alloc_no_inplace(void) {
    static int v = -1;
    if (v < 0) { const char * e = getenv("GGML_ALLOC_NO_INPLACE"); v = (e && e[0] && e[0] != '0') ? 1 : 0; }
    return v != 0;
}
// [2026-07-28 M3 wf=1 CODE-DEFAULT] Host-forced no-free-reuse, same pattern as
// g_ggml_protect_reuse below. The M3 wf=1 layer-major garble (L1 upper-half NaN,
// GPU-verified 07-28: targeted Q-pin alone is INSUFFICIENT - pinning one tensor
// just relocates the collision) is only reliably fixed by full no-free-reuse.
// llama-context enables this programmatically for the affected config so
// correctness does not ride on a hidden preset env (user directive: defaults
// must work). Env GGML_ALLOC_NO_FREE_REUSE still forces it on for A/B.
int g_ggml_force_no_free_reuse = 0;  // set by host (llama-context) per-config
void ggml_gallocr_set_no_free_reuse(int enable) { g_ggml_force_no_free_reuse = enable; }  // NO_FREE_REUSE_SET
static bool ggml_alloc_no_free_reuse(void) {
    static int v = -1;
    if (v < 0) { const char * e = getenv("GGML_ALLOC_NO_FREE_REUSE"); v = (e && e[0] && e[0] != '0') ? 1 : 0; }
    return v != 0 || g_ggml_force_no_free_reuse != 0;
}

// 2026-05-31 TARGETED no-free-reuse, GATED. The fix for the chunk5 free-reassign collision
// (above) is to NOT return the small attention-critical slots (norm/kv_cmpr/Kcur/Vcur/k_pe/
// attn_norm) to gallocr's reuse pool during the layer-major MLA multi-ubatch prefill, so a
// later writer cannot clobber a slot still needed across the repeated slice. This MUST be
// scoped to the prefill graph only: leaving it on for the single-token decode graph holds
// slots there too and inflates that buffer -> 215k decode OOM (token_embd GET_ROWS). So
// llama-context sets g_ggml_protect_reuse=1 around the layer-major prefill graph alloc and
// =0 (or leaves -1) for decode. Env LLAMA_PREFILL_PROTECT_REUSE overrides the gate (testing):
//   "1" = on (default attn set), "0" = off, any other string = protect that substring only.
int g_ggml_protect_reuse = -1;  // -1 = use env / off; 0 = off; 1 = on (set by host per-graph)
void ggml_gallocr_set_protect_reuse(int enable) { g_ggml_protect_reuse = enable; }  // PROTECT_REUSE_SET

bool ggml_op_can_inplace(enum ggml_op op) {
    switch (op) {
        case GGML_OP_FILL:
        case GGML_OP_SCALE:
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_UNARY:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_SILU_BACK:
        case GGML_OP_RMS_NORM:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_SOFT_MAX_BACK:
            return true;

        default:
            return false;
    }
}

static size_t aligned_offset(const void * buffer, size_t offset, size_t alignment) {
    assert(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)buffer + offset) % alignment)) % alignment;
    return offset + align;
}

// tallocr

struct ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer) {
    void * base = ggml_backend_buffer_get_base(buffer);
    size_t align = ggml_backend_buffer_get_alignment(buffer);

    assert(align && !(align & (align - 1))); // power of 2

    struct ggml_tallocr talloc = (struct ggml_tallocr) {
        /*.buffer    = */ buffer,
        /*.base      = */ base,
        /*.alignment = */ align,
        /*.offset    = */ aligned_offset(base, 0, align),
    };
    return talloc;
}

enum ggml_status ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor) {
    size_t size = ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
    size = GGML_PAD(size, talloc->alignment);

    if (talloc->offset + size > ggml_backend_buffer_get_size(talloc->buffer)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, size, ggml_backend_buffer_get_size(talloc->buffer) - talloc->offset);
        GGML_ABORT("not enough space in the buffer");
    }

    void * addr = (char *)ggml_backend_buffer_get_base(talloc->buffer) + talloc->offset;
    talloc->offset += size;

    assert(((uintptr_t)addr % talloc->alignment) == 0);

    return ggml_backend_tensor_alloc(talloc->buffer, tensor, addr);
}

// dynamic tensor allocator

#define GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
struct buffer_address {
    int chunk;     // index of a backend buffer
    size_t offset; // local memory offset within the buffer
};

static const struct buffer_address GGML_BUFFER_ADDRESS_INVALID = { -1, SIZE_MAX };

static bool ggml_buffer_address_less(struct buffer_address a, struct buffer_address b) {
    return a.chunk != b.chunk ? a.chunk < b.chunk : a.offset < b.offset;
}

struct free_block {
    size_t offset;
    size_t size;
};

struct tallocr_chunk {
    struct free_block free_blocks[MAX_FREE_BLOCKS];
    int n_free_blocks;
    size_t max_size;
};

struct ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_size;
    struct tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    int n_chunks;

#ifdef GGML_ALLOCATOR_DEBUG
    struct {
        const struct ggml_tensor * tensor;
        struct buffer_address addr;
    } allocated_tensors[1024];
#endif
};

static void ggml_dyn_tallocr_insert_block(struct tallocr_chunk * chunk, size_t offset, size_t size) {
    GGML_ASSERT(chunk->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    int insert_pos = 0;
    while (insert_pos < chunk->n_free_blocks && chunk->free_blocks[insert_pos].offset < offset) {
        insert_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    for (int i = chunk->n_free_blocks; i > insert_pos; i--) {
        chunk->free_blocks[i] = chunk->free_blocks[i-1];
    }
    // insert the new block
    chunk->free_blocks[insert_pos].offset = offset;
    chunk->free_blocks[insert_pos].size = size;
    chunk->n_free_blocks++;
}

static void ggml_dyn_tallocr_remove_block(struct tallocr_chunk * chunk, int idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    for (int i = idx; i < chunk->n_free_blocks; i++) {
        chunk->free_blocks[i] = chunk->free_blocks[i+1];
    }
    chunk->n_free_blocks--;
}

static int ggml_dyn_tallocr_new_chunk(struct ggml_dyn_tallocr * alloc, size_t min_size) {
    if (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        return -1;
    }
    struct tallocr_chunk * chunk = calloc(1, sizeof(struct tallocr_chunk));
    chunk->n_free_blocks = 1;
    chunk->free_blocks[0].offset = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_blocks[0].size = MAX(min_size, alloc->max_chunk_size);
    if (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_blocks[0].size = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    return alloc->n_chunks - 1;
}

#ifdef GGML_ALLOCATOR_DEBUG
static void add_allocated_tensor(struct ggml_dyn_tallocr * alloc, struct buffer_address addr, const struct ggml_tensor * tensor) {
    for (int i = 0; i < 1024; i++) {
        if (alloc->allocated_tensors[i].tensor == NULL) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            return;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
static void remove_allocated_tensor(struct ggml_dyn_tallocr * alloc, struct buffer_address addr, const struct ggml_tensor * tensor) {
    for (int i = 0; i < 1024; i++) {
        if (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.offset == addr.offset) {
            alloc->allocated_tensors[i].tensor = NULL;
            return;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#endif

static struct buffer_address ggml_dyn_tallocr_alloc(struct ggml_dyn_tallocr * alloc, size_t size, const struct ggml_tensor * tensor) {
    size = aligned_offset(NULL, size, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

    int best_fit_chunk = -1;
    int best_fit_block = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    for (int c = 0; c < alloc->n_chunks; ++c) {
        struct tallocr_chunk * chunk = alloc->chunks[c];
        size_t best_fit_size = SIZE_MAX;
        for (int i = 0; i < chunk->n_free_blocks - 1; i++) {
            struct free_block * block = &chunk->free_blocks[i];
            max_avail = MAX(max_avail, block->size);
            if (block->size >= size && block->size <= best_fit_size) {
                best_fit_chunk = c;
                best_fit_block = i;
                best_fit_size = block->size;
            }
        }
    }

    if (best_fit_block == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t best_reuse = INT64_MIN;
        for (int c = 0; c < alloc->n_chunks; ++c) {
            struct tallocr_chunk * chunk = alloc->chunks[c];
            if (chunk->n_free_blocks > 0) {
                struct free_block * block = &chunk->free_blocks[chunk->n_free_blocks - 1];
                max_avail = MAX(max_avail, block->size);
                int64_t reuse_factor = chunk->max_size - block->offset - size;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = best_reuse < 0 && reuse_factor > best_reuse;
                bool better_fit = reuse_factor >= 0 && reuse_factor < best_reuse;
                if (block->size >= size && (better_reuse || better_fit)) {
                    best_fit_chunk = c;
                    best_fit_block = chunk->n_free_blocks - 1;
                    best_reuse = reuse_factor;
                }
            }
        }
    }

    if (best_fit_block == -1) {
        // none of the existing chunks have enough space left
        best_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, size);
        best_fit_block = 0;
    }
    if (best_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, size, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    struct tallocr_chunk * chunk = alloc->chunks[best_fit_chunk];
    struct free_block    * block = &chunk->free_blocks[best_fit_block];
    struct buffer_address  addr  = {.chunk = best_fit_chunk, .offset = block->offset };
    block->offset += size;
    block->size -= size;
    if (block->size == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_block(chunk, best_fit_block);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", best_fit_block, addr.offset, addr.chunk);

#ifdef GGML_ALLOCATOR_DEBUG
    add_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.offset + size;
    if (cur_max > chunk->max_size) {
        // sort allocated_tensors by chunk/offset
        for (int i = 0; i < 1024; i++) {
            for (int j = i + 1; j < 1024; j++) {
                if (ggml_buffer_address_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    const struct ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    struct buffer_address tmp_addr = alloc->allocated_tensors[i].addr;
                    alloc->allocated_tensors[i].tensor = alloc->allocated_tensors[j].tensor;
                    alloc->allocated_tensors[i].addr = alloc->allocated_tensors[j].addr;
                    alloc->allocated_tensors[j].tensor = tmp_tensor;
                    alloc->allocated_tensors[j].addr = tmp_addr;
                }
            }
        }
        GGML_LOG_DEBUG("max_size[%d] = %.2f MB: tensors: ", addr.chunk, cur_max / 1024.0 / 1024.0);
        for (int i = 0; i < 1024; i++) {
            if (alloc->allocated_tensors[i].tensor) {
                GGML_LOG_DEBUG("%s [%d: %zx-%zx] (%.2f MB) ", alloc->allocated_tensors[i].tensor->name,
                    alloc->allocated_tensors[i].addr.chunk,
                    alloc->allocated_tensors[i].addr.offset,
                    alloc->allocated_tensors[i].addr.offset + ggml_nbytes(alloc->allocated_tensors[i].tensor),
                    ggml_nbytes(alloc->allocated_tensors[i].tensor) / 1024.0 / 1024.0);
            }
        }
        GGML_LOG_DEBUG("\n");
    }
#endif

    chunk->max_size = MAX(chunk->max_size, addr.offset + size);

    return addr;

    GGML_UNUSED(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
static void ggml_dyn_tallocr_free_bytes(struct ggml_dyn_tallocr * alloc, struct buffer_address addr, size_t size) {
    size = aligned_offset(NULL, size, alloc->alignment);

    struct tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    for (int i = 0; i < chunk->n_free_blocks; i++) {
        struct free_block * block = &chunk->free_blocks[i];
        // check if ptr is at the end of the block
        if (block->offset + block->size == addr.offset) {
            block->size += size;
            // check if we can merge with the next block
            if (i < chunk->n_free_blocks - 1) {
                struct free_block * next = &chunk->free_blocks[i+1];
                if (block->offset + block->size == next->offset) {
                    block->size += next->size;
                    ggml_dyn_tallocr_remove_block(chunk, i+1);
                }
            }
            return;
        }
        // check if ptr is at the beginning of the block
        if (addr.offset + size == block->offset) {
            block->offset = addr.offset;
            block->size += size;
            // check if we can merge with the previous block
            if (i > 0) {
                struct free_block * prev = &chunk->free_blocks[i-1];
                if (prev->offset + prev->size == block->offset) {
                    prev->size += block->size;
                    ggml_dyn_tallocr_remove_block(chunk, i);
                }
            }
            return;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_insert_block(chunk, addr.offset, size);
}

static void ggml_dyn_tallocr_reset(struct ggml_dyn_tallocr * alloc) {
    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        free(alloc->chunks[i]);
        alloc->chunks[i] = NULL;
    }
    alloc->n_chunks = 0;

#ifdef GGML_ALLOCATOR_DEBUG
    for (int i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = NULL;
    }
#endif
}

// Remove one fixed address range from a freshly-reset virtual allocator. This
// is deliberately narrower than disabling reuse: allocations before and after
// the range retain the normal free/reuse behavior.
static bool ggml_dyn_tallocr_reserve_bytes(
        struct ggml_dyn_tallocr * alloc,
        struct buffer_address addr,
        size_t size) {
    if (addr.chunk < 0 || addr.chunk >= GGML_VBUFFER_MAX_CHUNKS ||
            addr.offset == SIZE_MAX || size == 0) {
        return false;
    }

    size = aligned_offset(NULL, size, alloc->alignment);
    if (size > SIZE_MAX - addr.offset) {
        return false;
    }
    const size_t end = addr.offset + size;

    while (alloc->n_chunks <= addr.chunk) {
        const size_t min_size =
            alloc->n_chunks == addr.chunk ? end : 0;
        if (ggml_dyn_tallocr_new_chunk(alloc, min_size) < 0) {
            return false;
        }
    }

    struct tallocr_chunk * chunk = alloc->chunks[addr.chunk];
    for (int i = 0; i < chunk->n_free_blocks; ++i) {
        struct free_block * block = &chunk->free_blocks[i];
        if (block->size > SIZE_MAX - block->offset) {
            continue;
        }
        const size_t block_end = block->offset + block->size;
        if (addr.offset < block->offset || end > block_end) {
            continue;
        }

        if (addr.offset == block->offset && end == block_end) {
            ggml_dyn_tallocr_remove_block(chunk, i);
        } else if (addr.offset == block->offset) {
            block->offset = end;
            block->size = block_end - end;
        } else if (end == block_end) {
            block->size = addr.offset - block->offset;
        } else {
            if (chunk->n_free_blocks >= MAX_FREE_BLOCKS) {
                return false;
            }
            const size_t right_size = block_end - end;
            block->size = addr.offset - block->offset;
            ggml_dyn_tallocr_insert_block(chunk, end, right_size);
        }

        chunk->max_size = MAX(chunk->max_size, end);
        return true;
    }

    // A reserved range must have been wholly free. Hitting this path indicates
    // overlapping/duplicate reservations or a stale chunk offset.
    return false;
}

static struct ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_buffer_size) {
    struct ggml_dyn_tallocr * alloc = (struct ggml_dyn_tallocr *)malloc(sizeof(struct ggml_dyn_tallocr));

    *alloc = (struct ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_buffer_size, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {NULL},
        /*.n_chunks       = */ 0,
#ifdef GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#endif
    };

    ggml_dyn_tallocr_reset(alloc);

    return alloc;
}

static void ggml_dyn_tallocr_free(struct ggml_dyn_tallocr * alloc) {
    for (int i = 0; i < alloc->n_chunks; ++i) {
        free(alloc->chunks[i]);
    }
    free(alloc);
}

static size_t ggml_dyn_tallocr_max_size(struct ggml_dyn_tallocr * alloc, int chunk) {
    return chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_size : 0;
}


// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)

struct vbuffer {
    ggml_backend_buffer_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};

static void ggml_vbuffer_free(struct vbuffer * buf) {
    if (buf == NULL) {
        return;
    }
    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {
        ggml_backend_buffer_free(buf->chunks[i]);
    }
    free(buf);
}

static size_t ggml_vbuffer_chunk_size(struct vbuffer * buf, int chunk) {
    return buf->chunks[chunk] ? ggml_backend_buffer_get_size(buf->chunks[chunk]) : 0;
}

static size_t ggml_vbuffer_size(struct vbuffer * buf) {
    size_t size = 0;
    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        size += ggml_backend_buffer_get_size(buf->chunks[i]);
    }
    return size;
}

static struct vbuffer * ggml_vbuffer_alloc(ggml_backend_buffer_type_t buft, const struct ggml_dyn_tallocr * talloc, enum ggml_backend_buffer_usage usage) {
    struct vbuffer * buf = (struct vbuffer *)calloc(1, sizeof(struct vbuffer));
    if (buf == NULL) {
        return NULL;
    }

    for (int n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_size = talloc->chunks[n]->max_size;
        buf->chunks[n] = ggml_backend_buft_alloc_buffer(buft, chunk_size);
        if (buf->chunks[n] == NULL) {
            ggml_vbuffer_free(buf);
            return NULL;
        }
        ggml_backend_buffer_set_usage(buf->chunks[n], usage);
    }
    return buf;
}

static void ggml_vbuffer_tensor_alloc(struct vbuffer * buf, struct ggml_tensor * tensor, struct buffer_address buf_addr) {
    void * base = ggml_backend_buffer_get_base(buf->chunks[buf_addr.chunk]);
    void * addr = (char *)base + buf_addr.offset;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

static void ggml_vbuffer_reset(struct vbuffer * buf) {
    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_buffer_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

struct hash_node {
    int n_children;
    int n_views;
    int buffer_id;
    struct buffer_address addr;
    bool allocated;
};

struct tensor_alloc {
    int buffer_id;
    struct buffer_address addr;
    size_t size_max; // 0 = pre-allocated, unused, or view
};

struct leaf_alloc {
    struct tensor_alloc leaf;
};

struct node_alloc {
    struct tensor_alloc dst;
    struct tensor_alloc src[GGML_MAX_SRC];
};

struct ggml_alloc_output_guard_saved {
    int buffer_id;
    int chunk;
    size_t offset;
    size_t end_offset;
    uintptr_t begin;
    uintptr_t end;
    bool has_plan_range;
    bool has_pointer_range;
    char name[GGML_MAX_NAME];
};

struct ggml_gallocr_reserved_range {
    int buffer_id;
    struct buffer_address addr;
    size_t size;
    uintptr_t begin;
    uintptr_t end;
    const struct ggml_tensor * tensor;
    char name[GGML_MAX_NAME];
};

struct ggml_gallocr {
    ggml_backend_buffer_type_t * bufts; // [n_buffers]
    struct vbuffer ** buffers; // [n_buffers]
    struct ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    int n_buffers;
    const struct ggml_cgraph * reserved_graph; // identity of the most recent lifetime plan

    struct ggml_hash_set hash_set;
    struct hash_node * hash_values; // [hash_set.size]

    struct node_alloc * node_allocs; // [n_nodes]
    int n_nodes;

    struct leaf_alloc * leaf_allocs; // [n_leafs]
    int n_leafs;

    // Diagnostic-only cross-plan snapshot. Populated only when
    // LLAMA_ALLOC_OUTPUT_GUARD is enabled and a stable OUTPUT is observed.
    struct ggml_alloc_output_guard_saved * output_guard_saved;
    size_t output_guard_saved_count;
    size_t output_guard_saved_capacity;

    // Cross-plan occupied ranges. Placement uses offsets into the gallocr's
    // current vbuffer chunks, not external allocations; the tensor pointer and
    // absolute range are retained only for the opt-in audit.
    struct ggml_gallocr_reserved_range * reserved_ranges;
    size_t reserved_range_count;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)calloc(1, sizeof(struct ggml_gallocr));
    GGML_ASSERT(galloc != NULL);

    galloc->bufts = calloc(n_bufs, sizeof(ggml_backend_buffer_type_t));
    GGML_ASSERT(galloc->bufts != NULL);

    galloc->buffers = calloc(n_bufs, sizeof(struct vbuffer *));
    GGML_ASSERT(galloc->buffers != NULL);

    galloc->buf_tallocs = calloc(n_bufs, sizeof(struct ggml_dyn_tallocr *));
    GGML_ASSERT(galloc->buf_tallocs != NULL);

    for (int i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = NULL;

        // check if the same buffer type is used multiple times and reuse the same allocator
        for (int j = 0; j < i; j++) {
            if (bufts[i] == bufts[j]) {
                galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
                break;
            }
        }

        if (galloc->buf_tallocs[i] == NULL) {
            size_t alignment = ggml_backend_buft_get_alignment(bufts[i]);
            size_t max_size = ggml_backend_buft_get_max_size(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_size);
        }
    }
    galloc->n_buffers = n_bufs;

    return galloc;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft) {
    return ggml_gallocr_new_n(&buft, 1);
}

void ggml_gallocr_free(ggml_gallocr_t galloc) {
    if (galloc == NULL) {
        return;
    }

    for (int i = 0; i < galloc->n_buffers; i++) {
        if (galloc->buffers != NULL) {
            // skip if already freed
            bool freed = false;
            for (int j = 0; j < i; j++) {
                if (galloc->buffers[j] == galloc->buffers[i]) {
                    freed = true;
                    break;
                }
            }
            if (!freed) {
                ggml_vbuffer_free(galloc->buffers[i]);
            }
        }
        if (galloc->buf_tallocs != NULL) {
            // skip if already freed
            bool freed = false;
            for (int j = 0; j < i; j++) {
                if (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                    freed = true;
                    break;
                }
            }
            if (!freed) {
                ggml_dyn_tallocr_free(galloc->buf_tallocs[i]);
            }
        }
    }

    ggml_hash_set_free(&galloc->hash_set);
    free(galloc->hash_values);
    free(galloc->bufts);
    free(galloc->buffers);
    free(galloc->buf_tallocs);
    free(galloc->node_allocs);
    free(galloc->leaf_allocs);
    free(galloc->output_guard_saved);
    free(galloc->reserved_ranges);
    free(galloc);
}

typedef struct ggml_gallocr * ggml_gallocr_t;

static struct hash_node * ggml_gallocr_hash_get(ggml_gallocr_t galloc, struct ggml_tensor * t) {
    size_t i = ggml_hash_find_or_insert(&galloc->hash_set, t);
    return &galloc->hash_values[i];
}

// [ALLOC_OUTPUT_GUARD_V1]
//
// OUTPUT liveness is local to one gallocr plan. Layer-major streaming can
// rebuild graphs against the same galloc, and tensors already carrying data
// are external to the next plan. This diagnostic audits both the simulated
// reserve map and the actual data pointers applied to a graph. It is kept
// opt-in because an all-output/all-node comparison is intentionally verbose.
static bool ggml_alloc_output_guard_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("LLAMA_ALLOC_OUTPUT_GUARD");
        enabled = env && env[0] != '\0' && env[0] != '0';
    }
    return enabled != 0;
}

struct ggml_alloc_output_guard_range {
    int buffer_id;
    int chunk;
    size_t offset;
    size_t end;
    const struct ggml_tensor * root;
};

static const struct ggml_tensor * ggml_alloc_output_guard_root(
        const struct ggml_tensor * tensor, size_t * view_offset) {
    const struct ggml_tensor * root = tensor;
    size_t offset = 0;
    while (root && root->view_src) {
        if (root->view_offs > SIZE_MAX - offset) {
            return NULL;
        }
        offset += root->view_offs;
        root = root->view_src;
    }
    if (view_offset) {
        *view_offset = offset;
    }
    return root;
}

static bool ggml_alloc_output_guard_plan_range(
        ggml_gallocr_t galloc, const struct ggml_tensor * tensor,
        struct ggml_alloc_output_guard_range * range) {
    size_t view_offset = 0;
    const struct ggml_tensor * root =
        ggml_alloc_output_guard_root(tensor, &view_offset);
    if (!root || root->data != NULL || root->buffer != NULL) {
        return false;
    }

    struct hash_node * hn =
        ggml_gallocr_hash_get(galloc, (struct ggml_tensor *) root);
    if (hn->buffer_id < 0 || hn->buffer_id >= galloc->n_buffers ||
            hn->addr.chunk < 0 || hn->addr.offset == SIZE_MAX) {
        return false;
    }

    const size_t bytes = tensor->view_src ?
        ggml_nbytes(tensor) :
        ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], tensor);
    if (view_offset > SIZE_MAX - hn->addr.offset ||
            bytes > SIZE_MAX - (hn->addr.offset + view_offset)) {
        return false;
    }

    range->buffer_id = hn->buffer_id;
    range->chunk = hn->addr.chunk;
    range->offset = hn->addr.offset + view_offset;
    range->end = range->offset + bytes;
    range->root = root;
    return true;
}

static bool ggml_alloc_output_guard_intersects(
        ggml_gallocr_t galloc,
        const struct ggml_alloc_output_guard_range * a,
        const struct ggml_alloc_output_guard_range * b) {
    const bool same_arena =
        a->buffer_id == b->buffer_id ||
        galloc->buf_tallocs[a->buffer_id] ==
            galloc->buf_tallocs[b->buffer_id];
    return same_arena &&
           a->chunk == b->chunk &&
           a->offset < b->end &&
           b->offset < a->end;
}

static bool ggml_alloc_output_guard_is_stable(const struct ggml_tensor * tensor) {
    return tensor &&
        (tensor->flags & GGML_TENSOR_FLAG_OUTPUT) &&
        strstr(tensor->name, "indexer_topk_indices_stable") != NULL;
}

static struct ggml_alloc_output_guard_saved *
ggml_alloc_output_guard_saved_push(ggml_gallocr_t galloc) {
    if (galloc->output_guard_saved_count ==
            galloc->output_guard_saved_capacity) {
        size_t capacity = galloc->output_guard_saved_capacity ?
            galloc->output_guard_saved_capacity * 2 : 32;
        if (capacity < galloc->output_guard_saved_capacity ||
                capacity > SIZE_MAX /
                    sizeof(struct ggml_alloc_output_guard_saved)) {
            return NULL;
        }
        void * grown = realloc(
            galloc->output_guard_saved,
            capacity * sizeof(struct ggml_alloc_output_guard_saved));
        if (!grown) {
            return NULL;
        }
        galloc->output_guard_saved = grown;
        galloc->output_guard_saved_capacity = capacity;
    }
    struct ggml_alloc_output_guard_saved * saved =
        &galloc->output_guard_saved[galloc->output_guard_saved_count++];
    memset(saved, 0, sizeof(*saved));
    saved->buffer_id = -1;
    saved->chunk = -1;
    return saved;
}

static bool ggml_alloc_output_guard_pointer_plan_range(
        ggml_gallocr_t galloc, uintptr_t begin, uintptr_t end,
        int * buffer_id, int * chunk, size_t * offset, size_t * end_offset) {
    if (end <= begin) {
        return false;
    }
    for (int bi = 0; bi < galloc->n_buffers; ++bi) {
        struct vbuffer * vbuf = galloc->buffers[bi];
        if (!vbuf) {
            continue;
        }
        for (int ci = 0; ci < GGML_VBUFFER_MAX_CHUNKS; ++ci) {
            ggml_backend_buffer_t backend_buffer = vbuf->chunks[ci];
            if (!backend_buffer) {
                break;
            }
            const uintptr_t base =
                (uintptr_t) ggml_backend_buffer_get_base(backend_buffer);
            const size_t bytes = ggml_backend_buffer_get_size(backend_buffer);
            if (bytes > UINTPTR_MAX - base) {
                continue;
            }
            const uintptr_t buffer_end = base + bytes;
            if (begin >= base && end <= buffer_end) {
                *buffer_id = bi;
                *chunk = ci;
                *offset = (size_t) (begin - base);
                *end_offset = (size_t) (end - base);
                return true;
            }
        }
    }
    return false;
}

bool ggml_gallocr_set_reserved_tensors(
        ggml_gallocr_t galloc,
        struct ggml_tensor * const * tensors,
        size_t n_tensors) {
    GGML_ASSERT(galloc != NULL);
    galloc->reserved_graph = NULL;

    if (n_tensors == 0) {
        ggml_gallocr_clear_reserved_tensors(galloc);
        return true;
    }
    if (tensors == NULL ||
            n_tensors > SIZE_MAX / sizeof(struct ggml_gallocr_reserved_range)) {
        return false;
    }

    struct ggml_gallocr_reserved_range * ranges =
        calloc(n_tensors, sizeof(struct ggml_gallocr_reserved_range));
    if (!ranges) {
        return false;
    }

    size_t count = 0;
    for (size_t ti = 0; ti < n_tensors; ++ti) {
        struct ggml_tensor * tensor = tensors[ti];
        if (!tensor || !tensor->data || !tensor->buffer ||
                tensor->view_src || ggml_nbytes(tensor) == 0) {
            GGML_LOG_ERROR(
                    "%s: tensor %zu is not an allocated root tensor\n",
                    __func__, ti);
            free(ranges);
            return false;
        }

        const uintptr_t begin = (uintptr_t) tensor->data;
        const size_t logical_size = ggml_nbytes(tensor);
        if (logical_size > UINTPTR_MAX - begin) {
            free(ranges);
            return false;
        }

        int buffer_id = -1;
        int chunk = -1;
        size_t offset = 0;
        size_t logical_end_offset = 0;
        if (!ggml_alloc_output_guard_pointer_plan_range(
                    galloc, begin, begin + logical_size,
                    &buffer_id, &chunk, &offset, &logical_end_offset)) {
            GGML_LOG_ERROR(
                    "%s: tensor '%s' range [%p,%p) is not in the current gallocr buffers\n",
                    __func__, tensor->name[0] ? tensor->name : "(unnamed)",
                    tensor->data, (void *) (begin + logical_size));
            free(ranges);
            return false;
        }
        GGML_UNUSED(logical_end_offset);

        struct ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
        size_t alloc_size =
            ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor);
        alloc_size = aligned_offset(NULL, alloc_size, alloc->alignment);
        if (alloc_size == 0 || alloc_size > SIZE_MAX - offset) {
            free(ranges);
            return false;
        }

        const size_t end = offset + alloc_size;
        const size_t chunk_size =
            ggml_vbuffer_chunk_size(galloc->buffers[buffer_id], chunk);
        if (end > chunk_size) {
            GGML_LOG_ERROR(
                    "%s: tensor '%s' padded range exceeds gallocr chunk (%zu > %zu)\n",
                    __func__, tensor->name[0] ? tensor->name : "(unnamed)",
                    end, chunk_size);
            free(ranges);
            return false;
        }

        bool duplicate = false;
        for (size_t ri = 0; ri < count; ++ri) {
            const struct ggml_gallocr_reserved_range * prior = &ranges[ri];
            if (galloc->buf_tallocs[prior->buffer_id] != alloc ||
                    prior->addr.chunk != chunk) {
                continue;
            }
            const size_t prior_end = prior->addr.offset + prior->size;
            if (prior->addr.offset == offset && prior_end == end) {
                duplicate = true;
                break;
            }
            if (offset < prior_end && prior->addr.offset < end) {
                GGML_LOG_ERROR(
                        "%s: tensor '%s' reservation partially overlaps another reserved tensor\n",
                        __func__, tensor->name[0] ? tensor->name : "(unnamed)");
                free(ranges);
                return false;
            }
        }
        if (duplicate) {
            continue;
        }

        ranges[count++] = (struct ggml_gallocr_reserved_range) {
            /*.buffer_id =*/ buffer_id,
            /*.addr      =*/ { chunk, offset },
            /*.size      =*/ alloc_size,
            /*.begin     =*/ begin,
            /*.end       =*/ begin + logical_size,
            /*.tensor    =*/ tensor,
            /*.name      =*/ { 0 },
        };
        snprintf(ranges[count - 1].name, sizeof(ranges[count - 1].name),
                "%s", tensor->name[0] ? tensor->name : "(unnamed)");
    }

    if (count == 0) {
        free(ranges);
        return false;
    }

    free(galloc->reserved_ranges);
    galloc->reserved_ranges = ranges;
    galloc->reserved_range_count = count;
    return true;
}

void ggml_gallocr_clear_reserved_tensors(ggml_gallocr_t galloc) {
    if (!galloc) {
        return;
    }
    galloc->reserved_graph = NULL;
    free(galloc->reserved_ranges);
    galloc->reserved_ranges = NULL;
    galloc->reserved_range_count = 0;
}

static bool ggml_gallocr_reserved_audit_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("LLAMA_GLM_CONT_PLAN_AUDIT");
        enabled = env && env[0] != '\0' && env[0] != '0';
    }
    return enabled != 0;
}

static int ggml_gallocr_reserved_audit_layer(const char * name) {
    if (!name || !name[0]) {
        return -1;
    }
    const char * dash = strrchr(name, '-');
    if (!dash || !dash[1]) {
        return -1;
    }
    char * end = NULL;
    const long layer = strtol(dash + 1, &end, 10);
    return end && *end == '\0' && layer >= 0 && layer <= INT_MAX
        ? (int) layer : -1;
}

static bool ggml_gallocr_reserved_audit_is_original(
        const struct ggml_tensor * tensor,
        const struct ggml_gallocr_reserved_range * range) {
    const struct ggml_tensor * root = tensor;
    while (root && root->view_src) {
        root = root->view_src;
    }
    if (!root || root != range->tensor || !root->data) {
        return false;
    }
    const uintptr_t begin = (uintptr_t) root->data;
    const size_t bytes = ggml_nbytes(root);
    return bytes <= UINTPTR_MAX - begin &&
           begin == range->begin && begin + bytes == range->end;
}

static void ggml_gallocr_reserved_audit_graph(
        ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (!ggml_gallocr_reserved_audit_enabled() ||
            galloc->reserved_range_count == 0) {
        return;
    }

    int graph_layer = -1;
    bool mixed_layers = false;
    for (int i = 0; i < graph->n_nodes; ++i) {
        const int layer =
            ggml_gallocr_reserved_audit_layer(graph->nodes[i]->name);
        if (layer < 0) {
            continue;
        }
        if (graph_layer < 0) {
            graph_layer = layer;
        } else if (graph_layer != layer) {
            mixed_layers = true;
        }
    }
    if (mixed_layers) {
        graph_layer = -1;
    }

    int64_t planned_tensors = 0;
    int64_t overlaps = 0;
    const int n_total = graph->n_nodes + graph->n_leafs;
    for (int i = 0; i < n_total; ++i) {
        const struct ggml_tensor * tensor =
            i < graph->n_nodes
                ? graph->nodes[i]
                : graph->leafs[i - graph->n_nodes];
        if (!tensor || !tensor->data || ggml_nbytes(tensor) == 0) {
            continue;
        }
        const uintptr_t begin = (uintptr_t) tensor->data;
        const size_t bytes = ggml_nbytes(tensor);
        if (bytes > UINTPTR_MAX - begin) {
            continue;
        }
        const uintptr_t end = begin + bytes;
        ++planned_tensors;

        for (size_t ri = 0; ri < galloc->reserved_range_count; ++ri) {
            const struct ggml_gallocr_reserved_range * range =
                &galloc->reserved_ranges[ri];
            if (begin >= range->end || range->begin >= end ||
                    ggml_gallocr_reserved_audit_is_original(tensor, range)) {
                continue;
            }
            const uintptr_t overlap_begin = MAX(begin, range->begin);
            const uintptr_t overlap_end = MIN(end, range->end);
            const int tensor_layer =
                ggml_gallocr_reserved_audit_layer(tensor->name);
            ++overlaps;
            fprintf(stderr,
                    "*** GLM_CONT_PLAN_AUDIT_V1 OVERLAP: "
                    "il=%d stable='%s' planned='%s' "
                    "stable_range=[%p,%p) planned_range=[%p,%p) "
                    "overlap_bytes=%lld\n",
                    tensor_layer >= 0 ? tensor_layer : graph_layer,
                    range->name,
                    tensor->name[0] ? tensor->name : "(unnamed)",
                    (void *) range->begin, (void *) range->end,
                    (void *) begin, (void *) end,
                    (long long) (overlap_end - overlap_begin));
        }
    }

    fprintf(stderr,
            "*** GLM_CONT_PLAN_AUDIT_V1 PLAN_SUMMARY: "
            "il=%d planned_tensors=%lld overlaps=%lld status=%s\n",
            graph_layer, (long long) planned_tensors, (long long) overlaps,
            overlaps == 0 ? "CLEAN" : "COLLISION");
    fflush(stderr);
}

static size_t ggml_alloc_output_guard_capture_plan(
        ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    size_t captured = 0;
    bool has_planned_stable = false;
    for (int i = 0; i < graph->n_nodes; ++i) {
        struct ggml_alloc_output_guard_range range;
        if (ggml_alloc_output_guard_is_stable(graph->nodes[i]) &&
                ggml_alloc_output_guard_plan_range(
                    galloc, graph->nodes[i], &range)) {
            has_planned_stable = true;
            break;
        }
    }
    if (!has_planned_stable) {
        return 0;
    }

    galloc->output_guard_saved_count = 0;
    for (int i = 0; i < graph->n_nodes; ++i) {
        const struct ggml_tensor * output = graph->nodes[i];
        struct ggml_alloc_output_guard_range range;
        if (!ggml_alloc_output_guard_is_stable(output) ||
                !ggml_alloc_output_guard_plan_range(galloc, output, &range)) {
            continue;
        }
        struct ggml_alloc_output_guard_saved * saved =
            ggml_alloc_output_guard_saved_push(galloc);
        if (!saved) {
            fprintf(stderr,
                    "*** ALLOC_OUTPUT_GUARD_V1 ERROR: phase=snapshot_reserve "
                    "allocation_failed captured=%zu\n",
                    captured);
            break;
        }
        saved->buffer_id = range.buffer_id;
        saved->chunk = range.chunk;
        saved->offset = range.offset;
        saved->end_offset = range.end;
        saved->has_plan_range = true;
        snprintf(saved->name, sizeof(saved->name), "%s",
                 output->name[0] ? output->name : "(unnamed)");
        ++captured;
    }
    fprintf(stderr,
            "*** ALLOC_OUTPUT_GUARD_V1 SNAPSHOT: phase=reserve stable_outputs=%zu\n",
            captured);
    return captured;
}

static size_t ggml_alloc_output_guard_capture_applied(
        ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    bool has_stable = false;
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (ggml_alloc_output_guard_is_stable(graph->nodes[i]) &&
                graph->nodes[i]->data) {
            has_stable = true;
            break;
        }
    }
    if (!has_stable) {
        return 0;
    }

    galloc->output_guard_saved_count = 0;
    size_t captured = 0;
    for (int i = 0; i < graph->n_nodes; ++i) {
        const struct ggml_tensor * output = graph->nodes[i];
        if (!ggml_alloc_output_guard_is_stable(output) ||
                !output->data || ggml_nbytes(output) == 0) {
            continue;
        }
        const uintptr_t begin = (uintptr_t) output->data;
        const size_t bytes = ggml_nbytes(output);
        if (bytes > UINTPTR_MAX - begin) {
            continue;
        }
        struct ggml_alloc_output_guard_saved * saved =
            ggml_alloc_output_guard_saved_push(galloc);
        if (!saved) {
            fprintf(stderr,
                    "*** ALLOC_OUTPUT_GUARD_V1 ERROR: phase=snapshot_applied "
                    "allocation_failed captured=%zu\n",
                    captured);
            break;
        }
        saved->begin = begin;
        saved->end = begin + bytes;
        saved->has_pointer_range = true;
        saved->has_plan_range = ggml_alloc_output_guard_pointer_plan_range(
            galloc, saved->begin, saved->end,
            &saved->buffer_id, &saved->chunk,
            &saved->offset, &saved->end_offset);
        snprintf(saved->name, sizeof(saved->name), "%s",
                 output->name[0] ? output->name : "(unnamed)");
        ++captured;
    }
    fprintf(stderr,
            "*** ALLOC_OUTPUT_GUARD_V1 SNAPSHOT: phase=applied stable_outputs=%zu\n",
            captured);
    return captured;
}

static void ggml_alloc_output_guard_plan(
        ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (!ggml_alloc_output_guard_enabled()) {
        return;
    }

    static unsigned long long plan_seq = 0;
    const unsigned long long seq = ++plan_seq;
    size_t outputs = 0;
    size_t planned_outputs = 0;
    size_t external_outputs = 0;
    size_t intersections = 0;
    size_t hazards = 0;
    size_t stable_hazards = 0;
    size_t cross_plan_intersections = 0;
    size_t cross_plan_hazards = 0;
    bool current_planned_stable = false;
    for (int i = 0; i < graph->n_nodes; ++i) {
        struct ggml_alloc_output_guard_range range;
        if (ggml_alloc_output_guard_is_stable(graph->nodes[i]) &&
                ggml_alloc_output_guard_plan_range(
                    galloc, graph->nodes[i], &range)) {
            current_planned_stable = true;
            break;
        }
    }
    fprintf(stderr,
            "*** ALLOC_OUTPUT_GUARD_V1 ACTIVE: phase=reserve plan=%llu nodes=%d leafs=%d\n",
            seq, graph->n_nodes, graph->n_leafs);

    if (!current_planned_stable && galloc->output_guard_saved_count > 0) {
        for (size_t si = 0; si < galloc->output_guard_saved_count; ++si) {
            const struct ggml_alloc_output_guard_saved * saved =
                &galloc->output_guard_saved[si];
            if (!saved->has_plan_range) {
                continue;
            }
            for (int ni = 0; ni < graph->n_nodes; ++ni) {
                const struct ggml_tensor * node = graph->nodes[ni];
                struct ggml_alloc_output_guard_range node_range;
                if (!ggml_alloc_output_guard_plan_range(
                            galloc, node, &node_range) ||
                        (saved->buffer_id != node_range.buffer_id &&
                         galloc->buf_tallocs[saved->buffer_id] !=
                            galloc->buf_tallocs[node_range.buffer_id]) ||
                        saved->chunk != node_range.chunk ||
                        saved->offset >= node_range.end ||
                        node_range.offset >= saved->end_offset) {
                    continue;
                }
                const bool writer =
                    !ggml_is_view(node) && node->op != GGML_OP_NONE;
                ++cross_plan_intersections;
                cross_plan_hazards += writer;
                fprintf(stderr,
                        "*** ALLOC_OUTPUT_GUARD_V1 INTERSECTION: "
                        "phase=cross_plan_reserve plan=%llu saved_output='%s' "
                        "other_i=%d other='%s' buf=%d chunk=%d "
                        "output=[%zu,%zu) other=[%zu,%zu) writer=%d hazard=%d "
                        "historical_until_externalized=1\n",
                        seq, saved->name, ni,
                        node->name[0] ? node->name : "(unnamed)",
                        saved->buffer_id, saved->chunk,
                        saved->offset, saved->end_offset,
                        node_range.offset, node_range.end,
                        writer, writer);
            }
        }
    }

    for (int oi = 0; oi < graph->n_nodes; ++oi) {
        const struct ggml_tensor * output = graph->nodes[oi];
        if (!(output->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            continue;
        }
        ++outputs;
        struct ggml_alloc_output_guard_range out_range;
        if (!ggml_alloc_output_guard_plan_range(galloc, output, &out_range)) {
            ++external_outputs;
            fprintf(stderr,
                    "*** ALLOC_OUTPUT_GUARD_V1 OUTPUT_EXTERNAL: phase=reserve "
                    "plan=%llu index=%d name='%s' data=%p buffer=%p\n",
                    seq, oi, output->name[0] ? output->name : "(unnamed)",
                    output->data, (void *) output->buffer);
            continue;
        }
        ++planned_outputs;

        for (int ni = 0; ni < graph->n_nodes; ++ni) {
            if (ni == oi) {
                continue;
            }
            const struct ggml_tensor * node = graph->nodes[ni];
            struct ggml_alloc_output_guard_range node_range;
            if (!ggml_alloc_output_guard_plan_range(galloc, node, &node_range) ||
                    !ggml_alloc_output_guard_intersects(
                        galloc, &out_range, &node_range)) {
                continue;
            }

            const bool same_root = out_range.root == node_range.root;
            const bool later = ni > oi;
            const bool writer = !ggml_is_view(node) && node->op != GGML_OP_NONE;
            // Repeated node-range execution is not the full graph's single
            // post-order schedule. Any distinct writer sharing an OUTPUT range
            // can therefore execute after a value produced in an earlier
            // layer/ubatch, even when its full-graph index is lower.
            const bool hazard = writer && !same_root;
            const bool stable =
                strstr(output->name, "indexer_topk_indices_stable") != NULL;
            ++intersections;
            hazards += hazard;
            stable_hazards += hazard && stable;
            fprintf(stderr,
                    "*** ALLOC_OUTPUT_GUARD_V1 INTERSECTION: phase=reserve "
                    "plan=%llu output_i=%d output='%s' other_i=%d other='%s' "
                    "buf=%d chunk=%d output=[%zu,%zu) other=[%zu,%zu) "
                    "later=%d writer=%d same_root=%d hazard=%d stable=%d\n",
                    seq, oi, output->name[0] ? output->name : "(unnamed)",
                    ni, node->name[0] ? node->name : "(unnamed)",
                    out_range.buffer_id, out_range.chunk,
                    out_range.offset, out_range.end,
                    node_range.offset, node_range.end,
                    later, writer, same_root, hazard, stable);
        }
    }

    fprintf(stderr,
            "*** ALLOC_OUTPUT_GUARD_V1 SUMMARY: phase=reserve plan=%llu "
            "outputs=%zu planned=%zu external=%zu intersections=%zu hazards=%zu "
            "stable_hazards=%zu cross_plan_intersections=%zu "
            "cross_plan_hazards=%zu saved_outputs=%zu\n",
            seq, outputs, planned_outputs, external_outputs,
            intersections, hazards, stable_hazards,
            cross_plan_intersections, cross_plan_hazards,
            galloc->output_guard_saved_count);
    if (current_planned_stable) {
        ggml_alloc_output_guard_capture_plan(galloc, graph);
    }
    fflush(stderr);
}

static void ggml_alloc_output_guard_applied(
        ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (!ggml_alloc_output_guard_enabled()) {
        return;
    }

    static unsigned long long apply_seq = 0;
    const unsigned long long seq = ++apply_seq;
    size_t outputs = 0;
    size_t intersections = 0;
    size_t hazards = 0;
    size_t stable_hazards = 0;
    size_t cross_plan_intersections = 0;
    size_t cross_plan_hazards = 0;
    bool current_has_stable = false;
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (ggml_alloc_output_guard_is_stable(graph->nodes[i]) &&
                graph->nodes[i]->data) {
            current_has_stable = true;
            break;
        }
    }
    fprintf(stderr,
            "*** ALLOC_OUTPUT_GUARD_V1 ACTIVE: phase=applied graph=%llu nodes=%d leafs=%d\n",
            seq, graph->n_nodes, graph->n_leafs);

    if (!current_has_stable && galloc->output_guard_saved_count > 0) {
        for (size_t si = 0; si < galloc->output_guard_saved_count; ++si) {
            const struct ggml_alloc_output_guard_saved * saved =
                &galloc->output_guard_saved[si];
            if (!saved->has_pointer_range) {
                continue;
            }
            for (int ni = 0; ni < graph->n_nodes; ++ni) {
                const struct ggml_tensor * node = graph->nodes[ni];
                if (!node->data || ggml_nbytes(node) == 0) {
                    continue;
                }
                const uintptr_t node_begin = (uintptr_t) node->data;
                const size_t node_bytes = ggml_nbytes(node);
                if (node_bytes > UINTPTR_MAX - node_begin) {
                    continue;
                }
                const uintptr_t node_end = node_begin + node_bytes;
                if (saved->begin >= node_end || node_begin >= saved->end) {
                    continue;
                }
                const bool writer =
                    !ggml_is_view(node) && node->op != GGML_OP_NONE;
                ++cross_plan_intersections;
                cross_plan_hazards += writer;
                fprintf(stderr,
                        "*** ALLOC_OUTPUT_GUARD_V1 INTERSECTION: "
                        "phase=cross_plan_applied graph=%llu saved_output='%s' "
                        "other_i=%d other='%s' output=[%p,%p) other=[%p,%p) "
                        "writer=%d hazard=%d historical_until_externalized=1\n",
                        seq, saved->name, ni,
                        node->name[0] ? node->name : "(unnamed)",
                        (void *) saved->begin, (void *) saved->end,
                        (void *) node_begin, (void *) node_end,
                        writer, writer);
            }
        }
    }

    for (int oi = 0; oi < graph->n_nodes; ++oi) {
        const struct ggml_tensor * output = graph->nodes[oi];
        if (!(output->flags & GGML_TENSOR_FLAG_OUTPUT) ||
                !output->data || ggml_nbytes(output) == 0) {
            continue;
        }
        ++outputs;
        const uintptr_t out_begin = (uintptr_t) output->data;
        const size_t out_bytes = ggml_nbytes(output);
        if (out_bytes > UINTPTR_MAX - out_begin) {
            continue;
        }
        const uintptr_t out_end = out_begin + out_bytes;
        const struct ggml_tensor * output_root =
            ggml_alloc_output_guard_root(output, NULL);

        for (int ni = 0; ni < graph->n_nodes; ++ni) {
            if (ni == oi) {
                continue;
            }
            const struct ggml_tensor * node = graph->nodes[ni];
            if (!node->data || ggml_nbytes(node) == 0) {
                continue;
            }
            const uintptr_t node_begin = (uintptr_t) node->data;
            const size_t node_bytes = ggml_nbytes(node);
            if (node_bytes > UINTPTR_MAX - node_begin) {
                continue;
            }
            const uintptr_t node_end = node_begin + node_bytes;
            if (out_begin >= node_end || node_begin >= out_end) {
                continue;
            }

            const struct ggml_tensor * node_root =
                ggml_alloc_output_guard_root(node, NULL);
            const bool same_root = output_root == node_root;
            const bool later = ni > oi;
            const bool writer = !ggml_is_view(node) && node->op != GGML_OP_NONE;
            const bool hazard = writer && !same_root;
            const bool stable =
                strstr(output->name, "indexer_topk_indices_stable") != NULL;
            ++intersections;
            hazards += hazard;
            stable_hazards += hazard && stable;
            fprintf(stderr,
                    "*** ALLOC_OUTPUT_GUARD_V1 INTERSECTION: phase=applied "
                    "graph=%llu output_i=%d output='%s' other_i=%d other='%s' "
                    "output=[%p,%p) other=[%p,%p) later=%d writer=%d "
                    "same_root=%d hazard=%d stable=%d\n",
                    seq, oi, output->name[0] ? output->name : "(unnamed)",
                    ni, node->name[0] ? node->name : "(unnamed)",
                    (void *) out_begin, (void *) out_end,
                    (void *) node_begin, (void *) node_end,
                    later, writer, same_root, hazard, stable);
        }
    }

    fprintf(stderr,
            "*** ALLOC_OUTPUT_GUARD_V1 SUMMARY: phase=applied graph=%llu "
            "outputs=%zu intersections=%zu hazards=%zu stable_hazards=%zu "
            "cross_plan_intersections=%zu cross_plan_hazards=%zu "
            "saved_outputs=%zu\n",
            seq, outputs, intersections, hazards, stable_hazards,
            cross_plan_intersections, cross_plan_hazards,
            galloc->output_guard_saved_count);
    if (current_has_stable) {
        ggml_alloc_output_guard_capture_applied(galloc, graph);
    }
    fflush(stderr);
}

static bool ggml_gallocr_is_own(ggml_gallocr_t galloc, struct ggml_tensor * t) {
    return ggml_gallocr_hash_get(galloc, t)->allocated;
}

static bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, struct ggml_tensor * t) {
    return t->data != NULL // tensor data already set externally
        || t->buffer // tensor on external buffer (but not yet allocated)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
static void ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, struct ggml_tensor * node, struct ggml_tensor * parent) {
    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
    struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);

    size_t parent_size = ggml_backend_buft_get_alloc_size(galloc->bufts[p_hn->buffer_id], parent);
    size_t node_size = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], node);

    GGML_ASSERT(parent_size >= node_size);

    // note: we want after the freeing the chunks to continue to be aligned
    struct ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->buffer_id];
    parent_size = aligned_offset(NULL, parent_size, p_alloc->alignment);
    node_size = aligned_offset(NULL, node_size, p_alloc->alignment);

    if (parent_size > node_size) {
        struct buffer_address p_addr = p_hn->addr;
        p_addr.offset += node_size;
        size_t extra_size = parent_size - node_size;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_size, parent->name, node->name);
        ggml_dyn_tallocr_free_bytes(p_alloc, p_addr, extra_size);
    }
}

static void ggml_gallocr_allocate_node(ggml_gallocr_t galloc, struct ggml_tensor * node, int buffer_id) {
    GGML_ASSERT(buffer_id >= 0);
    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);

    if (!ggml_gallocr_is_allocated(galloc, node) && !ggml_is_view(node)) {
        hn->allocated = true;
        assert(hn->addr.offset == 0);

        // try to reuse a parent's buffer (inplace)
        const bool dsv41_rope_inplace = ggml_dsv41_rope_inplace_requested(node);
        const bool dsv41_masked_half_inplace = ggml_dsv41_masked_half_inplace_requested(node);
        if (!ggml_alloc_no_inplace() && (ggml_op_can_inplace(node->op) || ggml_dsv41_bf16_inplace_requested(node) || dsv41_rope_inplace || dsv41_masked_half_inplace)) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if ((dsv41_rope_inplace || dsv41_masked_half_inplace) && i != 0) continue;
                struct ggml_tensor * parent = node->src[i];
                if (parent == NULL) {
                    continue;
                }

                // Private masked-half full-view case. Views themselves are
                // not allocator-owned. Reuse only their owned root when this
                // is its sole complete view and that view's final reader.
                // Do not enable generic view reuse or relax any output rule.
                if (dsv41_masked_half_inplace && i == 0 && parent->view_src &&
                    !parent->view_src->view_src && parent->view_offs == 0 &&
                    ggml_is_contiguous(parent) && ggml_is_contiguous(parent->view_src) &&
                    parent->type == parent->view_src->type &&
                    ggml_nbytes(parent) == ggml_nbytes(parent->view_src) &&
                    ggml_are_same_layout(node, parent) &&
                    !(parent->flags & GGML_TENSOR_FLAG_OUTPUT) &&
                    !(parent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT) &&
                    ggml_gallocr_is_own(galloc, parent->view_src)) {
                    struct hash_node * view_hn = ggml_gallocr_hash_get(galloc, parent);
                    struct hash_node * root_hn = ggml_gallocr_hash_get(galloc, parent->view_src);
                    if (view_hn->n_children == 1 && view_hn->n_views == 0 &&
                        root_hn->n_children == 0 && root_hn->n_views == 1) {
                        hn->buffer_id = root_hn->buffer_id;
                        hn->addr = root_hn->addr;
                        root_hn->allocated = false;
                        ggml_gallocr_free_extra_space(galloc, node, parent->view_src);
                        return;
                    }
                }

                // if the node's data is external, then we cannot re-use it
                if (!ggml_gallocr_is_own(galloc, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, node->name, parent->data);
                    continue;
                }

                // outputs cannot be reused
                if (parent->flags & GGML_TENSOR_FLAG_OUTPUT || (parent->view_src != NULL && parent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", parent->name, node->name);
                    continue;
                }

                if (!ggml_are_same_layout(node, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", parent->name, node->name);
                    continue;
                }

                struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
                if (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    if (ggml_is_view(parent)) {
                        struct ggml_tensor * view_src = parent->view_src;
                        struct hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        if (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, node->name);
                            assert(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.offset == p_hn->addr.offset);
                            hn->buffer_id = p_hn->buffer_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, node, view_src);
                            return;
                        }
                    } else {
                        AT_PRINTF("reusing parent %s for %s\n", parent->name, node->name);
                        hn->buffer_id = p_hn->buffer_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, node, parent);
                        {  // ALLOC_OVERLAP in-place log: node inherits parent's addr (victim may be here)
                            static int aoi = -1;
                            if (aoi < 0) { const char * e = getenv("LLAMA_ALLOC_OVERLAP_DUMP"); aoi = (e && e[0] && e[0] != '0') ? 1 : 0; }
                            if (aoi && node->name[0]) {
                                size_t sz = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], node);
                                fprintf(stderr, "[ALLOC_OVERLAP] name=%-22s buf=%d chunk=%d off=%zu end=%zu size=%zu inplace_parent=%s\n",
                                        node->name, hn->buffer_id, (int) hn->addr.chunk, hn->addr.offset, hn->addr.offset + sz, sz, parent->name);
                                fflush(stderr);
                            }
                        }
                        return;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        struct ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
        ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
        size_t size = ggml_backend_buft_get_alloc_size(buft, node);
        hn->buffer_id = buffer_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, size, node);
        // 2026-05-31 ALLOC_OVERLAP_DUMP (env): print buffer_id/chunk/offset/size for the chunk5
        // collision suspects so an offline scan finds which buffer overlaps attn_norm-1/kv_cmpr_pe-1.
        // Marker ALLOC_OVERLAP for build-grep. Fires per gallocr reserve (per chunk); the largest
        // reserve (biggest self_kq_mask_cnv) = chunk5.
        {
            static int aod = -1;
            if (aod < 0) { const char * e = getenv("LLAMA_ALLOC_OVERLAP_DUMP"); aod = (e && e[0] && e[0] != '0') ? (e[0] - '0') : 0; }  // ALLOC_OVERLAP (1=filtered, 2=all)
            if (aod && node && node->name[0]) {
                const char * n = node->name;
                int hit = (aod >= 2) ||
                    strstr(n, "attn_norm-1") || strstr(n, "kv_cmpr_pe-1") || strstr(n, "Kcur-1") || strstr(n, "Vcur-1") ||
                    strstr(n, "self_kq_mask_cnv") || strstr(n, "ffn_moe") || strstr(n, "ffn_out-1") ||
                    strstr(n, "ffn_inp-1") || strstr(n, "l_out-1") || strstr(n, "norm-1") || strstr(n, "attn_out-1");
                if (hit) {
                    fprintf(stderr, "[ALLOC_OVERLAP] name=%-22s buf=%d chunk=%d off=%zu end=%zu size=%zu\n",
                            n, buffer_id, (int) hn->addr.chunk, hn->addr.offset, hn->addr.offset + size, size);
                    fflush(stderr);
                }
            }
        }
    }
}

static void ggml_gallocr_free_node(ggml_gallocr_t galloc, struct ggml_tensor * node) {
    {
        const char * dbg = getenv("SPLIT_DECODE_ALLOC_TRACE");
        if (dbg && node && node->name[0]) {
            const char * n = node->name;
            if (strstr(n, "-77") || strstr(n, "external_pl") || strstr(n, "-76") || strstr(n, "mega_out")) {
                fprintf(stderr, "[ALLOC_FREE] name='%s' data=%p\n", n, node->data);
                fflush(stderr);
            }
        }
    }
    // graph outputs are never freed
    if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", node->name);
        return;
    }

    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
    int buffer_id = hn->buffer_id;
    struct ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
    ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
    size_t size = ggml_backend_buft_get_alloc_size(buft, node);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, node->name, hn->addr.chunk, hn->addr.offset, size, alloc->chunks[hn->addr.chunk]->n_free_blocks);
#ifdef GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, node);
#endif

    // 2026-05-31 TARGETED no-free-reuse for the layer-major repeated-slice prefill collision.
    // The layer-major executor re-runs each layer's node slice PER UBATCH against a layout gallocr
    // froze for a SINGLE in-order pass. At chunk5's largest n_kv layout, a freed attention-critical
    // slot (norm/kv_cmpr/Kcur/Vcur — still needed across the repeated slice) is reassigned to a later
    // tensor that clobbers it. Skipping the free for these (small) tensors holds their slot so nothing
    // else reuses it — cheap (hundreds of MB), unlike global GGML_ALLOC_NO_FREE_REUSE (69GB OOM).
    // LLAMA_PREFILL_PROTECT_REUSE: unset/0 = off (production byte-identical); "1" = default attn set;
    // any other string = protect names containing THAT substring (targeted single-tensor test).
    {
        // env (read once): -2=unread, -1=unset, 0=force-off, 1=force-on
        static int env_state = -2; static const char * env_sub = NULL;
        if (env_state == -2) {
            const char * e = getenv("LLAMA_PREFILL_PROTECT_REUSE");
            if (!e || !e[0])      { env_state = -1; }
            else if (e[0] == '0') { env_state = 0; }
            else                  { env_state = 1; env_sub = (e[0] == '1' && e[1] == '\0') ? NULL : e; }
        }
        int pr; const char * pr_sub = NULL;
        if (env_state >= 0) { pr = env_state; pr_sub = env_sub; }            // env override
        else                { pr = (g_ggml_protect_reuse > 0) ? 1 : 0; }     // host gate (prefill-only)
        if (pr && node->name[0]) {
            const char * n = node->name;
            int prot = pr_sub
                ? (strstr(n, pr_sub) != NULL)
                : (strstr(n, "norm-") || strstr(n, "kv_cmpr") || strstr(n, "Kcur-") ||
                   strstr(n, "Vcur-") || strstr(n, "k_pe") || strstr(n, "attn_norm"));
            if (prot) {
                hn->allocated = false;   // mark freed for liveness, but DO NOT return bytes to the reuse pool
                return;
            }
        }
    }

    ggml_dyn_tallocr_free_bytes(alloc, hn->addr, size);
    hn->allocated = false;
}

static int get_node_buffer_id(const int * node_buffer_ids, int i) {
    return node_buffer_ids ? node_buffer_ids[i] : 0;
}

static void ggml_gallocr_alloc_graph_impl(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids) {
    // clear hash tables
    ggml_hash_set_reset(&galloc->hash_set);
    memset(galloc->hash_values, 0, sizeof(struct hash_node) * galloc->hash_set.size);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        ggml_gallocr_allocate_node(galloc, leaf, get_node_buffer_id(leaf_buffer_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        if (ggml_is_view(node) && node->op != GGML_OP_NONE) {
            struct ggml_tensor * view_src = node->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        if (node->flags & GGML_TENSOR_FLAG_INPUT) {
            ggml_gallocr_allocate_node(galloc, graph->nodes[i], get_node_buffer_id(node_buffer_ids, i));
        }

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                ggml_gallocr_allocate_node(galloc, src, get_node_buffer_id(node_buffer_ids, i));
            }
        }
    }

    // allocate tensors
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int buffer_id = get_node_buffer_id(node_buffer_ids, i);

        {
            const char * dbg = getenv("SPLIT_DECODE_ALLOC_TRACE");
            if (dbg && node && node->name[0]) {
                const char * n = node->name;
                if (strstr(n, "-77") || strstr(n, "external_pl") || strstr(n, "-76") || strstr(n, "mega_out")) {
                    fprintf(stderr, "[ALLOC_ALLOC] name='%s' op=%s\n", n, ggml_op_name(node->op));
                    fflush(stderr);
                }
            }
        }

        // allocate parents (only leafs need to be allocated at this point)
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            ggml_gallocr_allocate_node(galloc, parent, buffer_id);
        }

        // allocate node
        ggml_gallocr_allocate_node(galloc, node, buffer_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(node), node->name);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            AT_PRINTF("%s", parent->name);
            if (j < GGML_MAX_SRC - 1 && node->src[j + 1] != NULL) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                parent->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            if (p_hn->n_children == 0 && p_hn->n_views == 0) {
                if (ggml_is_view(parent)) {
                    struct ggml_tensor * view_src = parent->view_src;
                    struct hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    if (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        if (!ggml_alloc_no_free_reuse()) ggml_gallocr_free_node(galloc, view_src);
                    }
                }
                else if (p_hn->allocated && !ggml_alloc_no_free_reuse()) {
                    ggml_gallocr_free_node(galloc, parent);
                }
            }
            AT_PRINTF("\n");
        }
    }
    ggml_alloc_output_guard_plan(galloc, graph);
}

static bool ggml_gallocr_materialize_buffers(ggml_gallocr_t galloc, bool no_alloc) {
    // reallocate buffers if needed
    for (int i = 0; i < galloc->n_buffers; i++) {
        // if the buffer type is used multiple times, we reuse the same buffer
        for (int j = 0; j < i; j++) {
            if (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                galloc->buffers[i] = galloc->buffers[j];
                break;
            }
        }

        // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
        bool realloc = galloc->buffers[i] == NULL;
        size_t new_size = 0;
        for (int c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_size = galloc->buffers[i] ? ggml_vbuffer_chunk_size(galloc->buffers[i], c) : 0;
            size_t new_chunk_size = ggml_dyn_tallocr_max_size(galloc->buf_tallocs[i], c);
            new_size += new_chunk_size;
            if (new_chunk_size > cur_chunk_size) {
                realloc = true;
            }
        }
        if (realloc) {
#ifndef NDEBUG
            {
                size_t cur_size = galloc->buffers[i] ? ggml_vbuffer_size(galloc->buffers[i]) : 0;
                if (cur_size > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
                }
            }
#endif
            ggml_vbuffer_free(galloc->buffers[i]);
            if (no_alloc) {
                galloc->buffers[i] = NULL;
            } else {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                if (galloc->buffers[i] == NULL) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_size);
                    return false;
                }
            }
        }
    }

    return true;
}

static bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids, bool no_alloc) {
    galloc->reserved_graph = NULL;
    size_t min_hash_size = graph->n_nodes + graph->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_size += min_hash_size / 4;

    // initialize hash table
    if (galloc->hash_set.size < min_hash_size) {
        ggml_hash_set_free(&galloc->hash_set);
        galloc->hash_set = ggml_hash_set_new(min_hash_size);
        GGML_ASSERT(galloc->hash_set.keys != NULL);

        free(galloc->hash_values);
        galloc->hash_values = malloc(sizeof(struct hash_node) * galloc->hash_set.size);
        GGML_ASSERT(galloc->hash_values != NULL);
    }

    // reset allocators
    for (int i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // [GLM_CONT_PLAN_RESERVE_V1] Recreate the occupied holes captured from the
    // full graph before planning a later graph. Everything outside these exact
    // ranges remains available to normal gallocr lifetime reuse.
    for (size_t i = 0; i < galloc->reserved_range_count; ++i) {
        const struct ggml_gallocr_reserved_range * range =
            &galloc->reserved_ranges[i];
        if (range->buffer_id < 0 || range->buffer_id >= galloc->n_buffers ||
                !ggml_dyn_tallocr_reserve_bytes(
                    galloc->buf_tallocs[range->buffer_id],
                    range->addr, range->size)) {
            GGML_LOG_ERROR(
                    "%s: GLM_CONT_PLAN_RESERVE_V1 failed to reserve range %zu "
                    "(buffer=%d chunk=%d offset=%zu size=%zu)\n",
                    __func__, i, range->buffer_id, range->addr.chunk,
                    range->addr.offset, range->size);
            return false;
        }
    }

    // allocate in hash table
    ggml_gallocr_alloc_graph_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids);

    // set the node_allocs from the hash table
    if (galloc->n_nodes < graph->n_nodes) {
        free(galloc->node_allocs);
        galloc->node_allocs = calloc(graph->n_nodes, sizeof(struct node_alloc));
        GGML_ASSERT(galloc->node_allocs != NULL);
    }
    galloc->n_nodes = graph->n_nodes;
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];
        if (node->view_src || node->data) {
            node_alloc->dst.buffer_id = -1;
            node_alloc->dst.addr = GGML_BUFFER_ADDRESS_INVALID;
            node_alloc->dst.size_max = 0;
        } else {
            struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
            node_alloc->dst.buffer_id = hn->buffer_id;
            node_alloc->dst.addr = hn->addr;
            node_alloc->dst.size_max  = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], node);
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (!src || src->view_src || src->data) {
                node_alloc->src[j].buffer_id = -1;
                node_alloc->src[j].addr = GGML_BUFFER_ADDRESS_INVALID;
                node_alloc->src[j].size_max = 0;
            } else {
                struct hash_node * hn = ggml_gallocr_hash_get(galloc, src);
                node_alloc->src[j].buffer_id = hn->buffer_id;
                node_alloc->src[j].addr = hn->addr;
                node_alloc->src[j].size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], src);
            }
        }
    }
    if (galloc->n_leafs < graph->n_leafs) {
        free(galloc->leaf_allocs);
        galloc->leaf_allocs = calloc(graph->n_leafs, sizeof(galloc->leaf_allocs[0]));
        GGML_ASSERT(galloc->leaf_allocs != NULL);
    }
    galloc->n_leafs = graph->n_leafs;
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        struct hash_node * hn = ggml_gallocr_hash_get(galloc, leaf);
        if (leaf->view_src || leaf->data) {
            galloc->leaf_allocs[i].leaf.buffer_id = -1;
            galloc->leaf_allocs[i].leaf.addr = GGML_BUFFER_ADDRESS_INVALID;
            galloc->leaf_allocs[i].leaf.size_max = 0;
        } else {
            galloc->leaf_allocs[i].leaf.buffer_id = hn->buffer_id;
            galloc->leaf_allocs[i].leaf.addr = hn->addr;
            galloc->leaf_allocs[i].leaf.size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], leaf);
        }
    }

    const bool ok = ggml_gallocr_materialize_buffers(galloc, no_alloc);
    galloc->reserved_graph = ok && no_alloc ? graph : NULL;
    return ok;
}

void ggml_gallocr_reserve_n_size(
        ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids, size_t * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ true));
    for (int i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        for (int c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_size;
        }
    }
}

bool ggml_gallocr_materialize_reserved(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    // The caller admitted the size-only plan. Do not walk lifetimes again or
    // silently reserve another graph after that admission decision.
    if (!galloc || !graph || galloc->reserved_graph != graph) return false;
    galloc->reserved_graph = NULL;
    return ggml_gallocr_materialize_buffers(galloc, false);
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids) {
    return ggml_gallocr_reserve_n_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph *graph) {
    return ggml_gallocr_reserve_n(galloc, graph, NULL, NULL);
}

static void ggml_gallocr_init_tensor(ggml_gallocr_t galloc, struct ggml_tensor * tensor, struct tensor_alloc * tensor_alloc) {
    int buffer_id = tensor_alloc->buffer_id;
    assert(tensor->data || tensor->view_src || ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);

    if (tensor->view_src != NULL) {
        if (tensor->buffer == NULL) {
            assert(tensor_alloc->addr.offset == SIZE_MAX);
            if (tensor->view_src->buffer == NULL) {
                // this tensor was allocated without ggml-backend
                return;
            }
            ggml_backend_view_init(tensor);
        }
    } else {
        if (tensor->data == NULL) {
            assert(tensor_alloc->addr.offset != SIZE_MAX);
            assert(ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[buffer_id], tensor, tensor_alloc->addr);
        } else {
            if (tensor->buffer == NULL) {
                // this tensor was allocated without ggml-backend
                return;
            }
        }
    }
}

static bool ggml_gallocr_node_needs_realloc(ggml_gallocr_t galloc, struct ggml_tensor * node, struct tensor_alloc * talloc) {
    size_t node_size = 0;
    if (!node->data && !node->view_src) {
        // If we previously had data but don't now then reallocate
        if (talloc->buffer_id < 0) {
            return false;
        }
        node_size = ggml_backend_buft_get_alloc_size(galloc->bufts[talloc->buffer_id], node);
    }
    return talloc->size_max >= node_size;
}

static bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (galloc->n_nodes != graph->n_nodes) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#endif
        return true;
    }

    if (galloc->n_leafs != graph->n_leafs) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#endif
        return true;
    }

    // [2026-07-28 GLM-DSA L11/L3 host-AV FIX] needs_realloc validated nodes+srcs but NEVER the
    // leafs. The layer-major repeated-slice prefill reuses the reserved plan across per-layer
    // graphs; GLM-DSA's shared-cadence DENSE layers (il=3,7,11,...) carry a DIFFERENTLY-SIZED
    // leaf (full-n_kv mask vs the sparse layers' top_k=2048 scope) at the SAME leaf slot, so the
    // stale leaf_alloc passes the count checks, the Release build strips the guarding asserts in
    // ggml_gallocr_init_tensor, and ggml_vbuffer_tensor_alloc runs with a mismatched/unallocated
    // plan entry -> wild pointer -> host 0xc0000005 at ggml-base!0x1386a (WER, symbolized) or a
    // device-side IMA surfaced later at the DirectStorage fence (value=24). Mirror the node check
    // for every leaf so a stale plan forces a re-reserve instead. Same gallocr family as the M3
    // wf=1 free-then-reassign NaN (see GGML_ALLOC_NO_FREE_REUSE / PIN_MASK prior art).
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        struct leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];
        if (!ggml_gallocr_node_needs_realloc(galloc, leaf, &leaf_alloc->leaf)) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: leaf %s is not valid\n", __func__, leaf->name);
#endif
            return true;
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];

        if (!ggml_gallocr_node_needs_realloc(galloc, node, &node_alloc->dst)) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, node->name);
#endif
            return true;
        }

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            if (!ggml_gallocr_node_needs_realloc(galloc, src, &node_alloc->src[j])) {
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, node->name);
#endif
                return true;
            }
        }
    }

    return false;
}

bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    galloc->reserved_graph = NULL;
    if (ggml_gallocr_needs_realloc(galloc, graph)) {
        if (galloc->n_buffers == 1) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#endif
            if (!ggml_gallocr_reserve(galloc, graph)) {
                return false;
            }
        } else {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#endif
            return false;
        }
    }

    // reset buffers
    for (int i = 0; i < galloc->n_buffers; i++) {
        if (galloc->buffers[i] != NULL) {
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        struct leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];
        ggml_gallocr_init_tensor(galloc, leaf, &leaf_alloc->leaf);
    }
    // nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            ggml_gallocr_init_tensor(galloc, src, &node_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, node, &node_alloc->dst);
    }

    ggml_alloc_output_guard_applied(galloc, graph);
    ggml_gallocr_reserved_audit_graph(galloc, graph);
    return true;
}

size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id) {
    GGML_ASSERT(buffer_id >= 0 && buffer_id < galloc->n_buffers);

    if (galloc->buffers[buffer_id] == NULL) {
        return 0;
    }

    for (int i = 0; i < buffer_id; i++) {
        if (galloc->buffers[i] == galloc->buffers[buffer_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            return 0;
        }
    }

    return ggml_vbuffer_size(galloc->buffers[buffer_id]);
}

// utils

static void free_buffers(ggml_backend_buffer_t ** buffers, const size_t * n_buffers) {
    for (size_t i = 0; i < *n_buffers; i++) {
        ggml_backend_buffer_free((*buffers)[i]);
    }
    free(*buffers);
}

static bool alloc_tensor_range(struct ggml_context * ctx,
        struct ggml_tensor * first, struct ggml_tensor * last,
        ggml_backend_buffer_type_t buft, size_t size,
        ggml_backend_buffer_t ** buffers, size_t * n_buffers) {

    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), size);
        free_buffers(buffers, n_buffers);
        return false;
    }

    *buffers = realloc(*buffers, sizeof(ggml_backend_buffer_t) * (*n_buffers + 1));
    (*buffers)[(*n_buffers)++] = buffer;

    struct ggml_tallocr tallocr = ggml_tallocr_new(buffer);

    for (struct ggml_tensor * t = first; t != last; t = ggml_get_next_tensor(ctx, t)) {
        enum ggml_status status = GGML_STATUS_SUCCESS;
        if (t->data == NULL) {
            if (t->view_src == NULL) {
                status = ggml_tallocr_alloc(&tallocr, t);
            } else if (t->buffer == NULL) {
                status = ggml_backend_view_init(t);
            }
        } else {
            if (t->view_src != NULL && t->buffer == NULL) {
                // view of a pre-allocated tensor
                status = ggml_backend_view_init(t);
            }
        }
        if (status != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t->name);
            free_buffers(buffers, n_buffers);
            return false;
        }
    }

    return true;
}

static ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft_impl(
        struct ggml_context * ctx, ggml_backend_buffer_type_t buft, size_t * nbytes_total, bool no_alloc) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    size_t alignment = ggml_backend_buft_get_alignment(buft);
    size_t max_size = ggml_backend_buft_get_max_size(buft);

    ggml_backend_buffer_t * buffers = NULL;
    size_t n_buffers = 0;
    *nbytes_total = 0;

    size_t cur_buf_size = 0;
    struct ggml_tensor * first = ggml_get_first_tensor(ctx);
    for (struct ggml_tensor * t = first; t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        size_t this_size = 0;
        if (t->data == NULL && t->view_src == NULL) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        if (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            // allocate tensors in the current buffer
            if (!no_alloc && !alloc_tensor_range(ctx, first, t, buft, cur_buf_size, &buffers, &n_buffers)) {
                return NULL;
            }
            first = t;
            *nbytes_total += cur_buf_size;
            cur_buf_size = this_size;
        } else {
            cur_buf_size += this_size;
        }
    }

    // allocate remaining tensors
    if (cur_buf_size > 0) {
        *nbytes_total += cur_buf_size;
        if (!no_alloc && !alloc_tensor_range(ctx, first, NULL, buft, cur_buf_size, &buffers, &n_buffers)) {
            return NULL;
        }
    }

    if (no_alloc) {
        return NULL;
    }

    if (n_buffers == 0) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: all tensors in the context are already allocated\n", __func__);
#endif
        GGML_ASSERT(!buffers);
        return NULL;
    }

    ggml_backend_buffer_t buffer;
    if (n_buffers == 1) {
        buffer = buffers[0];
    } else {
        buffer = ggml_backend_multi_buffer_alloc_buffer(buffers, n_buffers);
    }
    if (buffers) {
        free(buffers); // can be NULL if context is empty or no_alloc
    }
    return buffer;
}

size_t ggml_backend_alloc_ctx_tensors_from_buft_size(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    size_t nbytes_total = 0;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft_impl(ctx, buft, &nbytes_total, /*no_alloc=*/ true);
    GGML_ASSERT(!buf);
    return nbytes_total;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    size_t nbytes_total = 0;
    return ggml_backend_alloc_ctx_tensors_from_buft_impl(ctx, buft, &nbytes_total, /*no_alloc =*/ false);
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors(struct ggml_context * ctx, ggml_backend_t backend) {
    return ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
}
