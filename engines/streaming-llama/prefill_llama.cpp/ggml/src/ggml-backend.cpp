// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif


// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));
        return size;
    }
    return ggml_nbytes(tensor);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    if (!buffer) {
        fprintf(stderr, "*** GGML_BACKEND: buffer_get_type called with NULL buffer\n");
        fflush(stderr);
    }
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

static bool ggml_backend_sched_skip_eval_tensor_copy(const struct ggml_tensor * tensor) {
    if (!tensor || !tensor->name) {
        return false;
    }

    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        return false;
    }

    return strncmp(tensor->name, "blk.", 4) == 0;
}

static bool ggml_backend_sched_is_mask_input(const struct ggml_tensor * tensor) {
    if (!tensor || !tensor->name) {
        return false;
    }
    return strstr(tensor->name, "kq_mask") != NULL;
}

static bool ggml_backend_sched_can_sync_input_copy_layout(const struct ggml_tensor * input, const struct ggml_tensor * input_cpy) {
    if (input == NULL || input_cpy == NULL) {
        return false;
    }

    if (!(input->flags & GGML_TENSOR_FLAG_INPUT)) {
        // KV-cache views (cache_k_l*/cache_v_l*) in the layer-major streaming path
        // are NOT INPUT-flagged but their cross-backend copy tensor can carry a stale
        // ne[1] (sized for an earlier ubatch's n_ubatch) while the live view spans the
        // padded n_batch -> ggml_are_same_layout asserts. They share type+nb, so syncing
        // the ne is the same fix as for INPUT tensors. Opt-in (off => unchanged behavior).
        static const bool kv_layout_sync = (getenv("LLAMA_PREFILL_KV_COPY_LAYOUT_SYNC") != NULL);
        const bool is_kv_cache = input->name &&
            (strncmp(input->name, "cache_k_l", 9) == 0 || strncmp(input->name, "cache_v_l", 9) == 0);
        if (!(kv_layout_sync && is_kv_cache)) {
            return false;
        }
    }

    if (input->type != input_cpy->type) {
        return false;
    }

    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (input->nb[i] != input_cpy->nb[i]) {
            return false;
        }
    }

    return true;
}

static void ggml_backend_sched_sync_input_copy_layout(struct ggml_tensor * input, struct ggml_tensor * input_cpy) {
    if (!ggml_backend_sched_can_sync_input_copy_layout(input, input_cpy)) {
        return;
    }

    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        input_cpy->ne[i] = input->ne[i];
    }
}

void ggml_backend_tensor_copy(struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (!ggml_are_same_layout(src, dst)) {
        fprintf(stderr, "%s: layout mismatch src=\"%s\" dst=\"%s\"\n",
                __func__,
                src && src->name ? src->name : "(null)",
                dst && dst->name ? dst->name : "(null)");
        fprintf(stderr, "%s: src type=%d ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n",
                __func__,
                src ? (int) src->type : -1,
                src ? (long long) src->ne[0] : 0, src ? (long long) src->ne[1] : 0,
                src ? (long long) src->ne[2] : 0, src ? (long long) src->ne[3] : 0,
                src ? src->nb[0] : 0, src ? src->nb[1] : 0, src ? src->nb[2] : 0, src ? src->nb[3] : 0);
        fprintf(stderr, "%s: dst type=%d ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n",
                __func__,
                dst ? (int) dst->type : -1,
                dst ? (long long) dst->ne[0] : 0, dst ? (long long) dst->ne[1] : 0,
                dst ? (long long) dst->ne[2] : 0, dst ? (long long) dst->ne[3] : 0,
                dst ? dst->nb[0] : 0, dst ? dst->nb[1] : 0, dst ? dst->nb[2] : 0, dst ? dst->nb[3] : 0);
        fflush(stderr);
    }
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
// 256: streaming prefill with ngl=0 routes all ops to GPU while weight
// and input leaf tensors remain on CPU.  With 64-layer models + KV cache
// index/mask inputs, the GPU split can accumulate 30+ unique cross-backend
// inputs.  At the old limit of 30 the scheduler created a same-backend
// GPU→GPU split mid-layer, breaking the streaming pipeline's per-layer
// weight remapping and causing illegal memory access.
#define GGML_SCHED_MAX_SPLIT_INPUTS 256
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor * inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_inputs;
    // graph view of this split
    struct ggml_cgraph graph;
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;
    uint64_t alloc_epoch;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor * graph_inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_graph_inputs;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;
    bool skip_eval_sync;
    bool skip_all_copies;
    bool skip_input_staging;  // skip INPUT-flagged copies (caller pre-stages via D2D)
    bool skip_mask_staging;   // skip mask tensor copies (GPU mask kernel generates on device)
    struct ggml_tensor * input_copy_skip_tensors[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_input_copy_skip_tensors;
    bool weight_copy_done;    // token_embd.weight already copied (skip subsequent copies)

    // Pre-compute callback: invoked after input copies, before graph compute.
    // Used by layer-major path to patch residual tensors after scheduler copies.
    void (*pre_compute_cb)(void * data);
    void * pre_compute_cb_data;

    int debug;

    // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int debug_realloc;
    int debug_graph_size;
    int debug_prev_graph_size;
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

static bool ggml_backend_sched_input_copy_is_explicitly_skipped(
        const ggml_backend_sched_t sched,
        const struct ggml_tensor * input) {
    for (int i = 0; i < sched->n_input_copy_skip_tensors; ++i) {
        if (sched->input_copy_skip_tensors[i] == input) {
            return true;
        }
    }
    return false;
}

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = tensor->src[i];
        if (src == NULL) {
            continue;
        }
        // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
        // not an ideal solution
        if (tensor->op != GGML_OP_ROPE && src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
            // check if a backend with higher prio wants to offload the op
            if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                for (int b = 0; b < src_backend_id; b++) {
                    if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                        SET_CAUSE(tensor, "1.off");
                        return b;
                    }
                }
            }
            SET_CAUSE(tensor, "1.wgt%d", i);
            return src_backend_id;
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_name(node->op), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                    // check if the split has too many inputs
                    // FIXME: count the number of inputs instead of only checking when full
                    if (split->n_inputs == GGML_SCHED_MAX_SPLIT_INPUTS) {
                        const size_t id = hash_id(src);
                        int src_backend_id = sched->hv_tensor_backend_ids[id];
                        bool supported = ggml_backend_sched_buffer_supported(sched, src, cur_backend_id);
                        if (src_backend_id != cur_backend_id && tensor_id_copy(id, cur_backend_id, 0) == NULL && !supported) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                if (need_new_split && node_backend_id == cur_backend_id && sched->debug) {
                    GGML_LOG_DEBUG("sched: same-backend split [%d->%d] at node %d: name=\"%s\" op=%s n_inputs=%d\n",
                            i_split, i_split + 1, i,
                            node->name ? node->name : "(null)",
                            ggml_op_name(node->op),
                            split->n_inputs);
                }
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src->flags & GGML_TENSOR_FLAG_INPUT && sched->n_copies > 1) {
                    if (tensor_id_copy(src_id, src_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[src_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy;
                            if (c == sched->cur_copy) {
                                tensor_copy = src; // use the original tensor as the current copy
                            } else {
                                tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                                ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            }
                            ggml_set_input(tensor_copy);
                            ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            tensor_id_copy(src_id, src_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_graph_inputs = sched->n_graph_inputs++;
                        GGML_ASSERT(n_graph_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        sched->graph_inputs[n_graph_inputs] = src;
                    }
                }

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            static const bool keep_input_copy_outputs = []() {
                                const char * env = getenv("LLAMA_SCHED_KEEP_INPUT_COPY_OUTPUT");
                                return env && env[0] != '\0' && env[0] != '0';
                            }();
                            const bool keep_small_input_copy =
                                keep_input_copy_outputs ||
                                ((src->flags & GGML_TENSOR_FLAG_INPUT) &&
                                 ggml_nbytes(src) <= 16ull * 1024ull * 1024ull);
                            if (sched->n_copies > 1 || keep_small_input_copy) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        for (int s = 0; s < sched->n_splits; s++) {
            const struct ggml_backend_sched_split * sp = &sched->splits[s];
            GGML_LOG_DEBUG("sched: split [%d/%d]: backend=%s nodes=[%d,%d) n_inputs=%d\n",
                    s, sched->n_splits,
                    sched->backends[sp->backend_id] ? ggml_backend_name(sched->backends[sp->backend_id]) : "(null)",
                    sp->i_start, sp->i_end, sp->n_inputs);
        }
    }

    // Layer-major backend placement inventory.  Keep this independent of the
    // general scheduler debug level so a production build can answer one very
    // specific question: which node range was assigned to which backend?  CPU
    // nodes also report whether the highest-priority backend supports the op;
    // this distinguishes an unsupported CUDA shape from buffer-affinity.
    static const bool dlm_split_dump = []() {
        const char * env = getenv("LLAMA_DLM_SPLIT_DUMP");
        return env && env[0] != '\0' && env[0] != '0';
    }();
    if (dlm_split_dump) {
        std::vector<int> backend_split_counts(sched->n_backends, 0);
        std::vector<int> backend_node_counts(sched->n_backends, 0);
        std::vector<int> cpu_op_counts(GGML_OP_COUNT, 0);

        fprintf(stderr, "*** DLM_SPLIT_DUMP: graph=%p nodes=%d splits=%d backends=%d\n",
                (void *) graph, graph->n_nodes, sched->n_splits, sched->n_backends);
        for (int s = 0; s < sched->n_splits; ++s) {
            const struct ggml_backend_sched_split * sp = &sched->splits[s];
            const int backend_id = sp->backend_id;
            int first = -1;
            int last = -1;
            std::vector<int> op_counts(GGML_OP_COUNT, 0);

            backend_split_counts[backend_id]++;
            for (int i = sp->i_start; i < sp->i_end; ++i) {
                const struct ggml_tensor * node = graph->nodes[i];
                if (ggml_is_view_op(node->op)) {
                    continue;
                }
                if (first < 0) {
                    first = i;
                }
                last = i;
                backend_node_counts[backend_id]++;
                if (node->op >= 0 && node->op < GGML_OP_COUNT) {
                    op_counts[node->op]++;
                    if (backend_id == sched->n_backends - 1) {
                        cpu_op_counts[node->op]++;
                    }
                }
            }

            const struct ggml_tensor * first_node = first >= 0 ? graph->nodes[first] : nullptr;
            const struct ggml_tensor * last_node  = last  >= 0 ? graph->nodes[last]  : nullptr;
            fprintf(stderr,
                    "*** DLM_SPLIT_DUMP: split=%d backend=%s nodes=[%d,%d) first=%d:%s/%s last=%d:%s/%s ops={",
                    s, ggml_backend_name(sched->backends[backend_id]), sp->i_start, sp->i_end,
                    first,
                    first_node && first_node->name[0] ? first_node->name : "(unnamed)",
                    first_node ? ggml_op_name(first_node->op) : "none",
                    last,
                    last_node && last_node->name[0] ? last_node->name : "(unnamed)",
                    last_node ? ggml_op_name(last_node->op) : "none");
            bool comma = false;
            for (int op = 0; op < GGML_OP_COUNT; ++op) {
                if (op_counts[op] == 0) {
                    continue;
                }
                fprintf(stderr, "%s%s:%d", comma ? "," : "", ggml_op_name((enum ggml_op) op), op_counts[op]);
                comma = true;
            }
            fprintf(stderr, "}\n");

            if (backend_id == sched->n_backends - 1 && sched->n_backends > 1) {
                for (int i = sp->i_start; i < sp->i_end; ++i) {
                    const struct ggml_tensor * node = graph->nodes[i];
                    if (ggml_is_view_op(node->op)) {
                        continue;
                    }
                    const bool gpu_support = ggml_backend_supports_op(sched->backends[0], node);
                    const ggml_backend_buffer_t dst_buf =
                            node->view_src && node->view_src->buffer ? node->view_src->buffer : node->buffer;
                    fprintf(stderr,
                            "*** DLM_SPLIT_DUMP_CPU: node=%d name=%s op=%s gpu0_support=%d "
                            "type=%s ne=[%lld,%lld,%lld,%lld] dst_buf=%s",
                            i, node->name[0] ? node->name : "(unnamed)", ggml_op_name(node->op),
                            gpu_support ? 1 : 0, ggml_type_name(node->type),
                            (long long) node->ne[0], (long long) node->ne[1],
                            (long long) node->ne[2], (long long) node->ne[3],
                            dst_buf ? ggml_backend_buffer_name(dst_buf) : "none");
                    for (int j = 0; j < GGML_MAX_SRC; ++j) {
                        const struct ggml_tensor * src = node->src[j];
                        if (!src) {
                            continue;
                        }
                        const ggml_backend_buffer_t src_buf =
                                src->view_src && src->view_src->buffer ? src->view_src->buffer : src->buffer;
                        fprintf(stderr, " src%d=%s%s[%lld,%lld,%lld,%lld]/%s", j,
                                ggml_type_name(src->type), ggml_is_contiguous(src) ? ":contig" : ":strided",
                                (long long) src->ne[0], (long long) src->ne[1],
                                (long long) src->ne[2], (long long) src->ne[3],
                                src_buf ? ggml_backend_buffer_name(src_buf) : "none");
                    }
                    fprintf(stderr, "\n");
                }
            }
        }

        for (int b = 0; b < sched->n_backends; ++b) {
            fprintf(stderr, "*** DLM_SPLIT_DUMP_SUMMARY: backend=%s splits=%d nodes=%d\n",
                    ggml_backend_name(sched->backends[b]), backend_split_counts[b], backend_node_counts[b]);
        }
        fprintf(stderr, "*** DLM_SPLIT_DUMP_CPU_OPS: ");
        bool comma = false;
        for (int op = 0; op < GGML_OP_COUNT; ++op) {
            if (cpu_op_counts[op] == 0) {
                continue;
            }
            fprintf(stderr, "%s%s:%d", comma ? "," : "", ggml_op_name((enum ggml_op) op), cpu_op_counts[op]);
            comma = true;
        }
        fprintf(stderr, "%s\n", comma ? "" : "none");
        fflush(stderr);
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + sched->n_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sched->n_copies;

    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]
    sched->debug_prev_graph_size = sched->debug_graph_size;
    sched->debug_graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        // Optimize this split of the graph. This needs to happen before we make graph_copy,
        // so they are in sync.
        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph);

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
        }
    }

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (sched->debug) {
        GGML_LOG_DEBUG("sched: alloc_splits backend_ids_changed=%d n_nodes=%d n_leafs=%d n_splits=%d\n",
                (int)backend_ids_changed, sched->graph.n_nodes, sched->graph.n_leafs, sched->n_splits);
    }
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
        if (sched->debug) {
            GGML_LOG_DEBUG("sched: triggering gallocr_reserve_n (backend_ids_changed=%d)\n",
                    (int)backend_ids_changed);
        }
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids);
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    struct ggml_backend_sched_split * splits = sched->splits;

    if (getenv("LLAMA_SPLIT_CAP_TRACE")) {
        fprintf(stderr, "[split_cap_trace] sched_compute_splits: n_splits=%d\n",
                sched->n_splits);
        fflush(stderr);
    }

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        if (getenv("LLAMA_SPLIT_CAP_TRACE")) {
            fprintf(stderr, "[split_cap_trace]   split %d: backend_id=%d n_inputs=%d\n",
                    split_id, split->backend_id, split->n_inputs);
            fflush(stderr);
        }
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // When skip_eval_sync is active (streaming prefill with weight fetcher),
        // weight tensors have been remapped to GPU by WF callbacks. The scheduler's
        // internal backend IDs are stale (still say CPU from ngl=0 reservation),
        // so it would try to copy every weight tensor CPU→GPU with sync each step.
        // Skip the entire input copy loop; WF guarantees GPU residency.
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

            // skip_eval_sync: layer weight tensors already on GPU (WF mode).
            // Only skip if: not INPUT-flagged AND the tensor name starts with "blk."
            // (i.e., it's a layer weight managed by the weight fetcher).
            // Non-layer weights (token_embd, output_norm) must still be copied.
            if (sched->skip_eval_sync && ggml_backend_sched_skip_eval_tensor_copy(input)) {
                continue;
            }

            if (sched->debug > 1) {
                GGML_LOG_DEBUG("sched: input [%d/%d]: name=\"%s\" nbytes=%zu src_buf=%s dst_buf=%s\n",
                        input_id, split->n_inputs,
                        input->name ? input->name : "(null)",
                        ggml_nbytes(input),
                        input->buffer ? ggml_backend_buffer_name(input->buffer) : "(null)",
                        input_cpy && input_cpy->buffer ? ggml_backend_buffer_name(input_cpy->buffer) : "(null)");
            }

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                ggml_backend_sched_sync_input_copy_layout(input, input_cpy);
                static const bool s_trace_kv_copy_a = (getenv("LLAMA_TRACE_KV_COPY") != NULL);
                if (s_trace_kv_copy_a && input->name && (strncmp(input->name, "cache_k_l", 9) == 0 || strncmp(input->name, "cache_v_l", 9) == 0)) {
                    fprintf(stderr, "ggml-backend copy site A: %s\n", input->name);
                    fflush(stderr);
                }
                ggml_backend_tensor_copy(input, input_cpy);
            } else {
                // wait for the split backend to finish using the input before overwriting it
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }

                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                ggml_tensor * node = split->graph.nodes[0];
                if (split->graph.n_nodes > 0 &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer) && (
                    (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)
                    //|| (node->src[1] == input_cpy && node->op == GGML_OP_ADD_ID) /* GGML_OP_ADD_ID weights are small and not worth splitting */
                    )) {

                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];

                    ggml_backend_synchronize(input_backend);

                    // get the ids
                    ggml_tensor * ids_tensor = node->src[2];
                    ggml_backend_t ids_backend = split_backend;

                    // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
                    // in that case, we use the original ids tensor
                    for (int i = input_id + 1; i < split->n_inputs; i++) {
                        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {
                            ids_tensor = split->inputs[i];
                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
                            break;
                        }
                    }

                    if (ids_tensor != prev_ids_tensor) {
                        ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
                        ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
                        ggml_backend_synchronize(ids_backend);

                        // find the used experts
                        used_ids.clear();
                        used_ids.resize(ggml_bitset_size(n_expert));
                        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                                int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                                GGML_ASSERT(id >= 0 && id < n_expert);
                                ggml_bitset_set(used_ids.data(), id);
                            }
                        }

                        prev_ids_tensor = ids_tensor;
                    }

                    // group consecutive experts and copy them together
                    auto copy_experts = [&](int32_t first_id, int32_t last_id) {
                        const size_t expert_offset = first_id * expert_size;
                        const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                        const size_t padding = std::min<size_t>(expert_size, 512);
                        const size_t padding_end = last_id < n_expert - 1 ? padding : 0;

                        ggml_backend_tensor_set_async(split_backend,
                            input_cpy,
                            (const uint8_t *)input->data + expert_offset, expert_offset,
                            // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
                            // this is necessary for MMQ in the CUDA backend
                            expert_size_copy + padding_end);
                    };

                    int id = 0;
                    while (!ggml_bitset_get(used_ids.data(), id)) {
                        id++;
                    }
                    int32_t first_id = id;
                    int32_t last_id = first_id;

                    for (++id; id < n_expert; ++id) {
                        if (!ggml_bitset_get(used_ids.data(), id)) {
                            continue;
                        }

                        if (id == last_id + 1) {
                            last_id = id;
                            continue;
                        }

                        copy_experts(first_id, last_id);

                        first_id = id;
                        last_id = id;
                    }
                    copy_experts(first_id, last_id);
                } else {
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
                        ggml_backend_synchronize(input_backend);
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                        }
                        ggml_backend_sched_sync_input_copy_layout(input, input_cpy);
                        static const bool s_trace_kv_copy_b = (getenv("LLAMA_TRACE_KV_COPY") != NULL);
                        if (s_trace_kv_copy_b && input->name && (strncmp(input->name, "cache_k_l", 9) == 0 || strncmp(input->name, "cache_v_l", 9) == 0)) {
                            fprintf(stderr, "ggml-backend copy site B: %s\n", input->name);
                            fflush(stderr);
                        }
                        ggml_backend_tensor_copy(input, input_cpy);
                    }
                }
            }
        }

        // Diagnostic: sync after input copies to catch any immediate CUDA errors
        if (sched->debug && split->n_inputs > 4) {
            ggml_backend_synchronize(split_backend);
        }

        const bool node_trace = std::getenv("LLAMA_PREFILL_NODE_TRACE") != nullptr;

        if (!sched->callback_eval && !node_trace) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            int subrange_idx = 0;
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                if (sched->debug > 1) {
                    const char * first_name = split->graph.nodes[j0]->name;
                    const char * last_name  = split->graph.nodes[j1]->name;
                    GGML_LOG_DEBUG("sched: compute subrange [split=%d sub=%d]: nodes [%d,%d) first=\"%s\" last=\"%s\"\n",
                            split_id, subrange_idx, j0, j1 + 1,
                            first_name ? first_name : "(null)",
                            last_name  ? last_name  : "(null)");
                }

                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                if (!sched->skip_eval_sync) {
                    ggml_backend_synchronize(split_backend);
                }

                if (sched->debug > 1) {
                    GGML_LOG_DEBUG("sched: compute subrange [split=%d sub=%d]: sync OK\n", split_id, subrange_idx);
                }

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
                subrange_idx++;
            }
        }

        // record the event of this copy
        if (split->n_inputs > 0) {
            if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    // Cap max_splits to prevent context_buffer explosion for models with
    // very large graphs (e.g. hybrid models whose graph_max_nodes uses
    // n_tokens * 40, yielding 80K-160K nodes).  The context_buffer is:
    //   max_splits * GGML_SCHED_MAX_SPLIT_INPUTS * 2 * sizeof(ggml_tensor)
    // With 80K splits this exceeds 14 GB, exhausting system RAM.
    // Real models have at most a few hundred splits even with multi-backend
    // graphs; 4096 provides ample headroom while keeping the buffer < 1 GB.
    const size_t ggml_sched_max_splits = std::min(graph_size, (size_t) 4096);
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    free(sched->splits);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
    ++sched->alloc_epoch;
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }
    ++sched->alloc_epoch;

    ggml_backend_sched_reset(sched);

    return true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;
    ++sched->alloc_epoch;

    return true;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    return ggml_backend_sched_compute_splits(sched);
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_copy_non_layer_inputs(ggml_backend_sched_t sched, const char * layer_prefix, int * n_copied) {
    GGML_ASSERT(sched);
    int count = 0;

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &sched->splits[split_id];
        int split_backend_id = split->backend_id;

        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

            if (!input_cpy || input_cpy == input) {
                continue;  // same backend, no copy needed
            }

            // Skip WF-managed tensors (layer weights + KV cache views).
            if (ggml_backend_sched_skip_eval_tensor_copy(input)) {
                continue;
            }

            // Skip caller-specified layer prefix.
            if (layer_prefix && input->name && strncmp(input->name, layer_prefix, strlen(layer_prefix)) == 0) {
                continue;
            }

            // skip_mask_staging: GPU generates the kq mask on-device; host copy vestigial. Mirrors sites D/E.
            if (sched->skip_mask_staging &&
                    ggml_backend_sched_is_mask_input(input)) {
                continue;
            }
            ggml_backend_synchronize(sched->backends[split_backend_id]);
            static const bool s_trace_kv_copy_c = (getenv("LLAMA_TRACE_KV_COPY") != NULL);
            if (s_trace_kv_copy_c && input->name && (strncmp(input->name, "cache_k_l", 9) == 0 || strncmp(input->name, "cache_v_l", 9) == 0)) {
                fprintf(stderr, "ggml-backend copy site C: %s\n", input->name);
                fflush(stderr);
            }
            ggml_backend_tensor_copy(input, input_cpy);
            count++;
        }
    }

    if (n_copied) {
        *n_copied = count;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

void ggml_backend_sched_set_eval_skip_sync(ggml_backend_sched_t sched, bool skip) {
    GGML_ASSERT(sched);
    sched->skip_eval_sync = skip;
}

static enum ggml_status ggml_backend_sched_compute_node_range_impl(
        ggml_backend_sched_t sched, int node_start, int node_end,
        int64_t * backend_compute_us, uint64_t * backend_compute_calls) {
    GGML_ASSERT(sched);
    GGML_ASSERT(sched->is_alloc && "graph must be allocated before computing node range");
    GGML_ASSERT(node_start >= 0 && node_end >= node_start);

    if (node_start == node_end) {
        return GGML_STATUS_SUCCESS;
    }

    struct ggml_backend_sched_split * splits = sched->splits;

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];

        // Check if this split overlaps with [node_start, node_end)
        if (split->i_end <= node_start || split->i_start >= node_end) {
            continue; // no overlap
        }

        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // Handle cross-backend input copies for this split.
        // skip_all_copies: caller handles data placement (pre-staged GPU inputs).
        // skip_eval_sync: weight tensors already on GPU (WF mode), copy INPUT-flagged only.
        if (!sched->skip_all_copies) {
            for (int input_id = 0; input_id < split->n_inputs; input_id++) {
                struct ggml_tensor * input = split->inputs[input_id];
                struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

                // skip_eval_sync: skip blk.* layer weights (WF-managed) AND
                // large non-INPUT weight tensors after first copy (e.g., token_embd 510 MB).
                if (sched->skip_eval_sync && ggml_backend_sched_skip_eval_tensor_copy(input)) {
                    continue;
                }

                // The caller has a pass-local device snapshot for this input
                // and restores its backend copy before each node range.
                if (ggml_backend_sched_input_copy_is_explicitly_skipped(sched, input)) {
                    continue;
                }

                // skip_input_staging: caller has pre-staged INPUT-flagged tensors
                // on GPU via D2D copy. Don't overwrite with stale CPU data.
                // EXCEPTION: mask tensors must still be copied — they're written
                // to CPU per-ubatch and the GPU buffer is reused by the allocator.
                if (sched->skip_input_staging && (input->flags & GGML_TENSOR_FLAG_INPUT)) {
                    bool is_mask = ggml_backend_sched_is_mask_input(input);
                    if (!is_mask) {
                        continue;  // skip non-mask INPUT tensors (staged via D2D)
                    }
                    if (is_mask && sched->skip_mask_staging) {
                        continue;  // skip mask copy — GPU kernel generates it directly
                    }
                }
                // skip_mask_staging is the signal that the GPU mask kernel generates the
                // mask on-device (use_gpu_mask), so the host copy of the kq mask is
                // vestigial. The skip above was incorrectly nested under skip_input_staging;
                // hoist it so the mask copy is skipped whenever skip_mask_staging is set.
                // This also avoids a layout-mismatch assert when the streaming mask leaf is
                // sized at constant-max (LLAMA_PREFILL_STREAMING_MASK_MAXKV) but its cached
                // cross-backend copy carries a per-ubatch (smaller) ne.
                if (sched->skip_mask_staging &&
                        ggml_backend_sched_is_mask_input(input)) {
                    continue;
                }

                // NOTE (2026-05-29): this per-input host drain was hypothesised to be the
                // dominant layer-major streaming cost and gated off via
                // LLAMA_PREFILL_STREAM_NO_COPY_DRAIN — REFUTED: removing it made prefill
                // SLOWER (104.8 vs 141 tok/s) AND corrupted state (decode "Invalid input
                // batch"). The drain is required for the cross-ubatch input-copy ordering;
                // it is NOT the bottleneck. The 5.2x gap vs stock is the 81-call layer-major
                // decomposition itself (vs stock's single graph_compute). Drain kept.
                ggml_backend_synchronize(split_backend);
                ggml_backend_sched_sync_input_copy_layout(input, input_cpy);
                static const bool s_trace_kv_copy_d = (getenv("LLAMA_TRACE_KV_COPY") != NULL);
                if (s_trace_kv_copy_d && input->name && (strncmp(input->name, "cache_k_l", 9) == 0 || strncmp(input->name, "cache_v_l", 9) == 0)) {
                    fprintf(stderr, "ggml-backend copy site D: %s\n", input->name);
                    fflush(stderr);
                }
                ggml_backend_tensor_copy(input, input_cpy);
                {
                    static const bool debug_kidxs_gpu = []() {
                        const char * env = getenv("LLAMA_PREFILL_DEBUG_KIDXS_GPU");
                        return env && env[0] != '\0' && env[0] != '0';
                    }();
                    static int debug_kidxs_gpu_count = 0;
                    if (debug_kidxs_gpu && input && input->name && strcmp(input->name, "k_idxs") == 0 &&
                            input_cpy && input_cpy->data && debug_kidxs_gpu_count < 80) {
                        const int64_t n64 = ggml_nelements(input_cpy);
                        const size_t n = n64 > 0 ? (size_t) n64 : 0;
                        if (n > 0 && (input_cpy->type == GGML_TYPE_I64 || input_cpy->type == GGML_TYPE_I32)) {
                            std::vector<int64_t> vals(n, 0);
                            if (input_cpy->type == GGML_TYPE_I64) {
                                ggml_backend_tensor_get(input_cpy, vals.data(), 0, n * sizeof(int64_t));
                            } else {
                                std::vector<int32_t> vals32(n, 0);
                                ggml_backend_tensor_get(input_cpy, vals32.data(), 0, n * sizeof(int32_t));
                                for (size_t i = 0; i < n; ++i) {
                                    vals[i] = vals32[i];
                                }
                            }

                            int64_t min_v = vals[0];
                            int64_t max_v = vals[0];
                            size_t backwards = 0;
                            for (size_t i = 0; i < n; ++i) {
                                min_v = std::min(min_v, vals[i]);
                                max_v = std::max(max_v, vals[i]);
                                if (i > 0 && vals[i] <= vals[i - 1]) {
                                    ++backwards;
                                }
                            }
                            std::vector<int64_t> sorted = vals;
                            std::sort(sorted.begin(), sorted.end());
                            size_t duplicates = 0;
                            size_t gaps = 0;
                            for (size_t i = 1; i < sorted.size(); ++i) {
                                if (sorted[i] == sorted[i - 1]) {
                                    ++duplicates;
                                } else if (sorted[i] > sorted[i - 1] + 1) {
                                    ++gaps;
                                }
                            }

                            fprintf(stderr,
                                    "*** KIDXS_GPU: copy=%d split=%d input=%p cpy=%p n=%zu min=%lld max=%lld backwards=%zu duplicates=%zu gaps=%zu first=[",
                                    debug_kidxs_gpu_count, split_id, (void *) input, (void *) input_cpy, n,
                                    (long long) min_v, (long long) max_v, backwards, duplicates, gaps);
                            const size_t first_n = std::min<size_t>(8, n);
                            for (size_t i = 0; i < first_n; ++i) {
                                fprintf(stderr, "%s%lld", i ? "," : "", (long long) vals[i]);
                            }
                            fprintf(stderr, "] last=[");
                            const size_t last_n = std::min<size_t>(8, n);
                            for (size_t i = n - last_n; i < n; ++i) {
                                fprintf(stderr, "%s%lld", i == n - last_n ? "" : ",", (long long) vals[i]);
                            }
                            fprintf(stderr, "]\n");
                            fflush(stderr);
                            ++debug_kidxs_gpu_count;
                        }
                    }
                }
            }
            // Mark weight copies done after first complete pass through all inputs
            if (sched->skip_eval_sync && !sched->weight_copy_done) {
                sched->weight_copy_done = true;
            }
        }

        // Pre-compute callback: runs AFTER input copies but BEFORE graph compute.
        // Used by layer-major path to D2D copy hidden state to residual tensor
        // addresses (which the scheduler's input copies may have overwritten).
        if (sched->pre_compute_cb) {
            sched->pre_compute_cb(sched->pre_compute_cb_data);
        }

        // Compute the overlapping portion of this split.
        // The split's graph view starts at split->i_start in the original graph.
        // We need to compute [max(node_start, split->i_start), min(node_end, split->i_end))
        // mapped to indices within the split's graph view.
        int range_start = (node_start > split->i_start) ? (node_start - split->i_start) : 0;
        int range_end   = (node_end < split->i_end)     ? (node_end   - split->i_start) : split->graph.n_nodes;

        if (range_start >= range_end) {
            continue;
        }

        struct ggml_cgraph gv = ggml_graph_view(&split->graph, range_start, range_end);

        const bool node_trace = std::getenv("LLAMA_PREFILL_NODE_TRACE") != nullptr;

        if (!sched->callback_eval && !node_trace) {
            const int64_t t_graph_us = backend_compute_us ? ggml_time_us() : 0;
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
            if (backend_compute_us) {
                *backend_compute_us += ggml_time_us() - t_graph_us;
                ++*backend_compute_calls;
            }
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
        } else {
            // Per-node compute with eval callback — used to find first diverging kernel
            for (int j = 0; j < gv.n_nodes; j++) {
                struct ggml_tensor * t = gv.nodes[j];
                if (node_trace) {
                    fprintf(stderr, "[node-trace] begin split=%d node=%d op=%s name=%s backend=%s\n",
                            split_id, split->i_start + range_start + j,
                            ggml_op_name(t->op), t->name, ggml_backend_name(split_backend));
                    fflush(stderr);
                }
                bool need = sched->callback_eval ? sched->callback_eval(t, true, sched->callback_eval_user_data) : false;
                struct ggml_cgraph gv1 = ggml_graph_view(&gv, j, j + 1);
                const int64_t t_graph_us = backend_compute_us ? ggml_time_us() : 0;
                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv1);
                if (backend_compute_us) {
                    *backend_compute_us += ggml_time_us() - t_graph_us;
                    ++*backend_compute_calls;
                }
                if (ec != GGML_STATUS_SUCCESS) { return ec; }
                // FA-node output dump (2026-05-30, LLAMA_FA_NODE_DUMP): read the __fattn__-{il} output
                // cksum RIGHT AFTER it computes (before downstream FFN reuses the buffer) to compare the
                // LOOP-path FA output vs the fast/C1 path (eval_cb hooks the same in graph_compute_per_layer).
                {
                    static const bool fa_dump = []() { const char * e = getenv("LLAMA_FA_NODE_DUMP"); return e && e[0] && e[0] != '0'; }();
                    static const int fa_dump_layer = []() { const char * e = getenv("LLAMA_FA_NODE_DUMP_LAYER"); return e && e[0] ? atoi(e) : -1; }();
                    bool fa_match = false;
                    if (fa_dump && t && t->name && strncmp(t->name, "__fattn__-", 10) == 0 && t->data) {
                        const int fil = atoi(t->name + 10);  // layer index after "__fattn__-"
                        fa_match = (fil == 0 || fil == 1 || fil == fa_dump_layer);
                    }
                    if (fa_match) {
                        ggml_backend_synchronize(split_backend);
                        auto ckof = [&](struct ggml_tensor * x) -> uint32_t {
                            if (!x || !x->data) return 0;
                            const size_t xn = ggml_nbytes(x); const size_t m = xn < 1048576 ? xn : 1048576;
                            std::vector<uint8_t> xb(m); ggml_backend_tensor_get(x, xb.data(), 0, m);
                            uint32_t c = 0; for (size_t i = 0; i < m; i++) c = c*31 + xb[i]; return c;
                        };
                        const size_t nb = ggml_nbytes(t);
                        fprintf(stderr, "*** FA_NODE_DUMP[loop] %s out_ck=0x%08x ne=[%lld,%lld,%lld] | Q=0x%08x K=0x%08x V=0x%08x M=0x%08x | Kne1=%lld Mne0=%lld\n",
                                t->name, ckof(t), (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
                                ckof(t->src[0]), ckof(t->src[1]), ckof(t->src[2]), ckof(t->src[3]),
                                (long long)(t->src[1]?t->src[1]->ne[1]:-1), (long long)(t->src[3]?t->src[3]->ne[0]:-1));
                        // Per-query-row value dump (ub-size sensitivity test): row 0 attends only to K[0]
                        // (=V[0], must be ub-invariant); row 100/3000 have long causal K-loops. Row q ==
                        // position q in both single-ub and multi-ub-u0 => directly comparable across ub.
                        if (t->type == GGML_TYPE_F32) {
                            auto rowval = [&](long long q) {
                                if (q >= t->ne[2]) { fprintf(stderr, "  row%-5lld [oob]\n", q); return; }
                                float v[4] = {0,0,0,0};
                                ggml_backend_tensor_get(t, v, (size_t)q * t->nb[2], sizeof(v));
                                fprintf(stderr, "  row%-5lld out[0..3]= % .6f % .6f % .6f % .6f\n", q, v[0], v[1], v[2], v[3]);
                            };
                            rowval(0); rowval(100); rowval(3000);
                        }
                        // K-source raw-byte dump (seed-location pin): is K[row] bit-identical across
                        // ub? If YES => the 1-ULP seed is FA-internal (ne11); if NO => it's the
                        // projection GEMM (cuBLAS/MMQ N-dependence). K = src[1], row stride nb[1].
                        if (t->src[1] && t->src[1]->data) {
                            struct ggml_tensor * K = t->src[1];
                            auto krow = [&](long long q) {
                                if (q >= K->ne[1]) { fprintf(stderr, "  K row%-5lld [oob ne1=%lld]\n", q, (long long)K->ne[1]); return; }
                                unsigned char b[16] = {0};
                                ggml_backend_tensor_get(K, b, (size_t)q * K->nb[1], sizeof(b));
                                fprintf(stderr, "  K row%-5lld [16B]= %02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x\n",
                                    q, b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],b[8],b[9],b[10],b[11],b[12],b[13],b[14],b[15]);
                            };
                            krow(0); krow(100); krow(2950);
                        }
                        (void) nb; fflush(stderr);
                    }
                }
                if (node_trace) {
                    ggml_backend_synchronize(split_backend);
                    fprintf(stderr, "[node-trace] end split=%d node=%d op=%s name=%s backend=%s\n",
                            split_id, split->i_start + range_start + j,
                            ggml_op_name(t->op), t->name, ggml_backend_name(split_backend));
                    fflush(stderr);
                }
                if (need) {
                    if (!sched->callback_eval(t, false, sched->callback_eval_user_data)) { break; }
                }
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_backend_sched_compute_node_range(
        ggml_backend_sched_t sched, int node_start, int node_end) {
    return ggml_backend_sched_compute_node_range_impl(sched, node_start, node_end, nullptr, nullptr);
}

enum ggml_status ggml_backend_sched_compute_node_range_profiled(
        ggml_backend_sched_t sched, int node_start, int node_end,
        int64_t * backend_compute_us, uint64_t * backend_compute_calls) {
    GGML_ASSERT(backend_compute_us);
    GGML_ASSERT(backend_compute_calls);
    *backend_compute_us = 0;
    *backend_compute_calls = 0;
    return ggml_backend_sched_compute_node_range_impl(
            sched, node_start, node_end, backend_compute_us, backend_compute_calls);
}

void ggml_backend_sched_set_pre_compute_cb(ggml_backend_sched_t sched,
                                            void (*cb)(void *), void * data) {
    GGML_ASSERT(sched);
    sched->pre_compute_cb = cb;
    sched->pre_compute_cb_data = data;
}

void ggml_backend_sched_set_skip_all_copies(ggml_backend_sched_t sched, bool skip) {
    GGML_ASSERT(sched);
    sched->skip_all_copies = skip;
}

void ggml_backend_sched_set_skip_input_staging(ggml_backend_sched_t sched, bool skip) {
    GGML_ASSERT(sched);
    sched->skip_input_staging = skip;
}

void ggml_backend_sched_set_skip_mask_staging(ggml_backend_sched_t sched, bool skip) {
    GGML_ASSERT(sched);
    sched->skip_mask_staging = skip;
}

void ggml_backend_sched_set_input_copy_skip_tensors(
        ggml_backend_sched_t sched,
        struct ggml_tensor * const * tensors,
        size_t n_tensors) {
    GGML_ASSERT(sched);
    GGML_ASSERT(n_tensors <= GGML_SCHED_MAX_SPLIT_INPUTS);
    GGML_ASSERT(n_tensors == 0 || tensors != nullptr);

    sched->n_input_copy_skip_tensors = (int) n_tensors;
    for (size_t i = 0; i < n_tensors; ++i) {
        sched->input_copy_skip_tensors[i] = tensors[i];
    }
    for (size_t i = n_tensors; i < GGML_SCHED_MAX_SPLIT_INPUTS; ++i) {
        sched->input_copy_skip_tensors[i] = nullptr;
    }
}

void ggml_backend_sched_stage_inputs(ggml_backend_sched_t sched, int node_start, int node_end) {
    GGML_ASSERT(sched);
    GGML_ASSERT(sched->is_alloc && "graph must be allocated before staging inputs");
    GGML_ASSERT(node_start >= 0 && node_end >= node_start);

    if (node_start == node_end) {
        return;
    }

    struct ggml_backend_sched_split * splits = sched->splits;

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];

        if (split->i_end <= node_start || split->i_start >= node_end) {
            continue;
        }

        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

            if (sched->skip_eval_sync && ggml_backend_sched_skip_eval_tensor_copy(input)) {
                continue;
            }

            // skip_mask_staging: GPU mask kernel generates the kq mask on-device, so the
            // host copy is vestigial. Skip it (also avoids the constant-max streaming mask
            // leaf vs per-ubatch copy-buffer layout-mismatch assert). Mirrors site D.
            if (sched->skip_mask_staging &&
                    ggml_backend_sched_is_mask_input(input)) {
                continue;
            }
            ggml_backend_synchronize(split_backend);
            ggml_backend_sched_sync_input_copy_layout(input, input_cpy);
            static const bool s_trace_kv_copy_e = (getenv("LLAMA_TRACE_KV_COPY") != NULL);
            if (s_trace_kv_copy_e && input->name && (strncmp(input->name, "cache_k_l", 9) == 0 || strncmp(input->name, "cache_v_l", 9) == 0)) {
                fprintf(stderr, "ggml-backend copy site E: %s\n", input->name);
                fflush(stderr);
            }
            ggml_backend_tensor_copy(input, input_cpy);
        }
    }
}

void ggml_backend_sched_get_mask_gpu_ptrs(ggml_backend_sched_t sched, int node_start, int node_end,
                                           void ** out_ptrs, int * out_n_kv, int * out_n_tok,
                                           int * out_count, int max_out) {
    GGML_ASSERT(sched);
    GGML_ASSERT(sched->is_alloc);
    *out_count = 0;
    if (node_start == node_end) return;

    struct ggml_backend_sched_split * splits = sched->splits;
    struct ggml_tensor * visited[32];
    int n_visited = 0;

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        if (split->i_end <= node_start || split->i_start >= node_end) continue;

        int split_backend_id = split->backend_id;

        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            struct ggml_tensor * input = split->inputs[input_id];

            if (!(input->flags & GGML_TENSOR_FLAG_INPUT)) continue;
            const char * name = input->name ? input->name : "";
            if (!strstr(name, "kq_mask")) continue;

            bool already = false;
            for (int v = 0; v < n_visited; v++) {
                if (visited[v] == input) { already = true; break; }
            }
            if (already) continue;
            if (n_visited < 32) visited[n_visited++] = input;

            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

            int idx = *out_count;
            if (idx >= max_out) return;
            out_ptrs[idx]  = input_cpy->data;
            out_n_kv[idx]  = (int)input_cpy->ne[0];
            out_n_tok[idx] = (int)input_cpy->ne[1];
            (*out_count)++;
        }
    }
}

void ggml_backend_sched_visit_input_copies(ggml_backend_sched_t sched,
                                            int node_start, int node_end,
                                            ggml_backend_sched_input_visitor visitor,
                                            void * user_data) {
    GGML_ASSERT(sched);
    GGML_ASSERT(sched->is_alloc && "graph must be allocated before visiting input copies");

    struct ggml_backend_sched_split * splits = sched->splits;

    // Track visited tensors to avoid duplicates across splits
    struct ggml_tensor * visited[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_visited = 0;

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];

        if (split->i_end <= node_start || split->i_start >= node_end) {
            continue;
        }

        int split_backend_id = split->backend_id;

        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            struct ggml_tensor * input = split->inputs[input_id];

            // Only INPUT-flagged tensors (skip weight tensors)
            if (!(input->flags & GGML_TENSOR_FLAG_INPUT)) {
                continue;
            }

            // Dedup
            bool already_visited = false;
            for (int v = 0; v < n_visited; v++) {
                if (visited[v] == input) { already_visited = true; break; }
            }
            if (already_visited) continue;
            if (n_visited < GGML_SCHED_MAX_SPLIT_INPUTS) {
                visited[n_visited++] = input;
            }

            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);
            if (input_cpy && input_cpy->data) {
                if (!visitor(input_cpy, user_data)) {
                    return;
                }
            }
        }
    }
}

void ggml_backend_sched_visit_input_pairs(ggml_backend_sched_t sched,
                                           int node_start, int node_end,
                                           ggml_backend_sched_input_pair_visitor visitor,
                                           void * user_data) {
    GGML_ASSERT(sched);
    GGML_ASSERT(sched->is_alloc && "graph must be allocated before visiting input pairs");

    struct ggml_tensor * visited_inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    struct ggml_tensor * visited_copies[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_visited = 0;

    for (int split_id = 0; split_id < sched->n_splits; ++split_id) {
        struct ggml_backend_sched_split * split = &sched->splits[split_id];
        if (split->i_end <= node_start || split->i_start >= node_end) {
            continue;
        }

        const int split_backend_id = split->backend_id;
        for (int input_id = 0; input_id < split->n_inputs; ++input_id) {
            struct ggml_tensor * input = split->inputs[input_id];
            if (!(input->flags & GGML_TENSOR_FLAG_INPUT)) {
                continue;
            }

            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);
            if (!input_cpy || !input_cpy->data) {
                continue;
            }

            bool already_visited = false;
            for (int v = 0; v < n_visited; ++v) {
                if (visited_inputs[v] == input && visited_copies[v] == input_cpy) {
                    already_visited = true;
                    break;
                }
            }
            if (already_visited) {
                continue;
            }
            if (n_visited < GGML_SCHED_MAX_SPLIT_INPUTS) {
                visited_inputs[n_visited] = input;
                visited_copies[n_visited] = input_cpy;
                ++n_visited;
            }

            if (!visitor(input, input_cpy, user_data)) {
                return;
            }
        }
    }
}

void ggml_backend_sched_visit_mask_input_copies(ggml_backend_sched_t sched,
                                                int node_start, int node_end,
                                                ggml_backend_sched_input_visitor visitor,
                                                void * user_data) {
    GGML_ASSERT(sched);
    GGML_ASSERT(sched->is_alloc && "graph must be allocated before visiting input copies");

    struct ggml_backend_sched_split * splits = sched->splits;

    struct ggml_tensor * visited[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_visited = 0;

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];

        if (split->i_end <= node_start || split->i_start >= node_end) {
            continue;
        }

        int split_backend_id = split->backend_id;

        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            struct ggml_tensor * input = split->inputs[input_id];

            if (!(input->flags & GGML_TENSOR_FLAG_INPUT)) {
                continue;
            }

            const char * name = input->name ? input->name : "";
            if (!strstr(name, "kq_mask")) {
                continue;
            }

            bool already_visited = false;
            for (int v = 0; v < n_visited; v++) {
                if (visited[v] == input) {
                    already_visited = true;
                    break;
                }
            }
            if (already_visited) {
                continue;
            }
            if (n_visited < GGML_SCHED_MAX_SPLIT_INPUTS) {
                visited[n_visited++] = input;
            }

            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);
            if (input_cpy && input_cpy->data) {
                if (!visitor(input_cpy, user_data)) {
                    return;
                }
            }
        }
    }
}

void ggml_backend_sched_visit_mask_input_pairs(ggml_backend_sched_t sched,
                                               int node_start, int node_end,
                                               ggml_backend_sched_input_pair_visitor visitor,
                                               void * user_data) {
    GGML_ASSERT(sched);
    GGML_ASSERT(sched->is_alloc && "graph must be allocated before visiting input pairs");

    struct ggml_backend_sched_split * splits = sched->splits;

    struct ggml_tensor * visited[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_visited = 0;

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];

        if (split->i_end <= node_start || split->i_start >= node_end) {
            continue;
        }

        int split_backend_id = split->backend_id;

        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            struct ggml_tensor * input = split->inputs[input_id];

            if (!(input->flags & GGML_TENSOR_FLAG_INPUT)) {
                continue;
            }

            const char * name = input->name ? input->name : "";
            if (!strstr(name, "kq_mask")) {
                continue;
            }

            bool already_visited = false;
            for (int v = 0; v < n_visited; v++) {
                if (visited[v] == input) {
                    already_visited = true;
                    break;
                }
            }
            if (already_visited) {
                continue;
            }
            if (n_visited < GGML_SCHED_MAX_SPLIT_INPUTS) {
                visited[n_visited++] = input;
            }

            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);
            if (input && input_cpy && input_cpy->data) {
                if (!visitor(input, input_cpy, user_data)) {
                    return;
                }
            }
        }
    }
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

uint64_t ggml_backend_sched_get_alloc_epoch(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->alloc_epoch;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

bool ggml_backend_sched_set_reserved_tensors(
        ggml_backend_sched_t sched,
        struct ggml_tensor * const * tensors,
        size_t n_tensors) {
    GGML_ASSERT(sched);
    return ggml_gallocr_set_reserved_tensors(
            sched->galloc, tensors, n_tensors);
}

void ggml_backend_sched_clear_reserved_tensors(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ggml_gallocr_clear_reserved_tensors(sched->galloc);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    if (tensor->buffer != NULL) {
        fprintf(stderr, "*** GGML_BACKEND_TENSOR_ALLOC: tensor=%s op=%s ne=[%lld,%lld,%lld,%lld] type=%d already has buffer=%p data=%p\n",
                tensor->name ? tensor->name : "(null)",
                ggml_op_name(tensor->op),
                (long long) tensor->ne[0], (long long) tensor->ne[1],
                (long long) tensor->ne[2], (long long) tensor->ne[3],
                (int) tensor->type, (void*)tensor->buffer, tensor->data);
        fflush(stderr);
    }
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT((char *)addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
                (char *)ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
