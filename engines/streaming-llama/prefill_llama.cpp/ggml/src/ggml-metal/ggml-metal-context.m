#import "ggml-metal-context.h"
#import "ggml-metal.h"  // for struct ggml_metal_gated_segment (Path A Phase D.2)

#import "ggml-impl.h"
#import "ggml-backend-impl.h"

#import "ggml-metal-impl.h"
#import "ggml-metal-common.h"
#import "ggml-metal-ops.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// max number of MTLCommandBuffer used to submit a graph for processing
#define GGML_METAL_MAX_COMMAND_BUFFERS 8

struct ggml_metal_command_buffer {
    id<MTLCommandBuffer> obj;
};

struct ggml_metal {
    char name[128];

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;

    ggml_metal_event_t ev_cpy; // for async copies

    dispatch_queue_t d_queue;

    // additional, inference-time compiled pipelines
    ggml_metal_pipelines_t pipelines_ext;

    bool use_fusion;
    bool use_concurrency;
    bool use_graph_optimize;

    int debug_graph;
    int debug_fusion;

    // how many times a given op was fused
    uint64_t fuse_cnt[GGML_OP_COUNT];

    // capture state
    bool capture_next_compute;
    bool capture_started;

    id<MTLCaptureScope> capture_scope;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers
    int n_nodes_0;      // number of nodes submitted by the main thread
    int n_nodes_1;      // remaining number of nodes submitted by the n_cb threads
    int n_nodes_per_cb;

    struct ggml_cgraph * gf;

    // the callback given to the thread pool
    void (^encode_async)(size_t ith);

    // n_cb command buffers + 1 used by the main thread
    struct ggml_metal_command_buffer cmd_bufs[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // extra command buffers for things like getting, setting and copying tensors
    NSMutableArray * cmd_bufs_ext;

    // the last command buffer queued into the Metal queue with operations relevant to the current Metal backend
    id<MTLCommandBuffer> cmd_buf_last;

    // abort ggml_metal_graph_compute if callback returns true
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;
};

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);

#if TARGET_OS_OSX && !GGML_METAL_NDEBUG
    // Show all the Metal device instances in the system
    NSArray * devices = MTLCopyAllDevices();
    for (id<MTLDevice> device in devices) {
        GGML_LOG_INFO("%s: found device: %s\n", __func__, [[device name] UTF8String]);
    }
    [devices release]; // since it was created by a *Copy* C method
#endif

    // init context
    ggml_metal_t res = calloc(1, sizeof(struct ggml_metal));

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);

    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, [[device name] UTF8String]);

    // TODO: would it be better to have one queue for the backend and one queue for the device?
    //       the graph encoders and async ops would use the backend queue while the sync ops would use the device queue?
    //res->queue = [device newCommandQueue]; [TAG_QUEUE_PER_BACKEND]
    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
    if (queue == nil) {
        GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
        return NULL;
    }

    res->dev = dev;
    res->lib = ggml_metal_device_get_library(dev);
    if (res->lib == NULL) {
        GGML_LOG_WARN("%s: the device does not have a precompiled Metal library - this is unexpected\n", __func__);
        GGML_LOG_WARN("%s: will try to compile it on the fly\n", __func__);

        res->lib = ggml_metal_library_init(dev);
        if (res->lib == NULL) {
            GGML_LOG_ERROR("%s: error: failed to initialize the Metal library\n", __func__);

            free(res);

            return NULL;
        }
    }

    res->ev_cpy = ggml_metal_device_event_init(dev);

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    snprintf(res->name, sizeof(res->name), "%s", props_dev->name);

    res->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

    res->use_fusion      = getenv("GGML_METAL_FUSION_DISABLE") == nil;
    res->use_concurrency = getenv("GGML_METAL_CONCURRENCY_DISABLE") == nil;

    {
        const char * val = getenv("GGML_METAL_GRAPH_DEBUG");
        res->debug_graph = val ? atoi(val) : 0;
    }

    {
        const char * val = getenv("GGML_METAL_FUSION_DEBUG");
        res->debug_fusion = val ? atoi(val) : 0;
    }

    res->use_graph_optimize = true;

    if (getenv("GGML_METAL_GRAPH_OPTIMIZE_DISABLE") != NULL) {
        res->use_graph_optimize = false;
    }

    memset(res->fuse_cnt, 0, sizeof(res->fuse_cnt));

    GGML_LOG_INFO("%s: use fusion         = %s\n", __func__, res->use_fusion         ? "true" : "false");
    GGML_LOG_INFO("%s: use concurrency    = %s\n", __func__, res->use_concurrency    ? "true" : "false");
    GGML_LOG_INFO("%s: use graph optimize = %s\n", __func__, res->use_graph_optimize ? "true" : "false");

    res->capture_next_compute = false;
    res->capture_started = false;
    res->capture_scope = nil;

    res->gf = nil;
    res->encode_async = nil;
    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        res->cmd_bufs[i].obj = nil;
    }

    res->cmd_bufs_ext = [[NSMutableArray alloc] init];

    res->cmd_buf_last = nil;

    res->pipelines_ext = ggml_metal_pipelines_init();

    return res;
}

void ggml_metal_free(ggml_metal_t ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        if (ctx->cmd_bufs[i].obj) {
            [ctx->cmd_bufs[i].obj release];
        }
    }

    for (int i = 0; i < (int) ctx->cmd_bufs_ext.count; ++i) {
        if (ctx->cmd_bufs_ext[i]) {
            [ctx->cmd_bufs_ext[i] release];
        }
    }

    [ctx->cmd_bufs_ext removeAllObjects];
    [ctx->cmd_bufs_ext release];

    if (ctx->pipelines_ext) {
        ggml_metal_pipelines_free(ctx->pipelines_ext);
        ctx->pipelines_ext = nil;
    }

    if (ctx->debug_fusion > 0) {
        GGML_LOG_DEBUG("%s: fusion stats:\n", __func__);
        for (int i = 0; i < GGML_OP_COUNT; i++) {
            if (ctx->fuse_cnt[i] == 0) {
                continue;
            }

            // note: cannot use ggml_log here
            GGML_LOG_DEBUG("%s: - %s: %" PRIu64 "\n", __func__, ggml_op_name((enum ggml_op) i), ctx->fuse_cnt[i]);
        }
    }

    Block_release(ctx->encode_async);

    //[ctx->queue release]; // [TAG_QUEUE_PER_BACKEND]

    dispatch_release(ctx->d_queue);

    ggml_metal_device_event_free(ctx->dev, ctx->ev_cpy);

    free(ctx);
}

const char * ggml_metal_get_name(ggml_metal_t ctx) {
    return ctx->name;
}

void ggml_metal_synchronize(ggml_metal_t ctx) {
    // wait for any backend operations to finish
    if (ctx->cmd_buf_last) {
        [ctx->cmd_buf_last waitUntilCompleted];
        // Phase 49: GPU timing probe — print actual GPU execution time after wait.
        {
            static bool gpu_timing = false;
            static dispatch_once_t gpu_timing_once;
            dispatch_once(&gpu_timing_once, ^{ gpu_timing = (getenv("GGML_METAL_GPU_TIMING") != NULL); });
            if (gpu_timing) {
                id<MTLCommandBuffer> cb = ctx->cmd_buf_last;
                const double gpu_ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
                fprintf(stderr, "*** METAL_GPU_TIME: gpu=%.2f ms status=%d\n",
                        gpu_ms, (int)[cb status]);
                fflush(stderr);
            }
        }
        ctx->cmd_buf_last = nil;
    }

    // check status of all command buffers
    {
        const int n_cb = ctx->n_cb;

        for (int cb_idx = 0; cb_idx <= n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;
            if (!cmd_buf) {
                continue;
            }

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, cb_idx, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }
                GGML_ABORT("fatal error");
            }
        }
    }

    // release any completed extra command buffers
    if (ctx->cmd_bufs_ext.count > 0) {
        for (size_t i = 0; i < ctx->cmd_bufs_ext.count; ++i) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs_ext[i];

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, (int) i, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }
                GGML_ABORT("fatal error");
            }

            [cmd_buf release];
        }

        [ctx->cmd_bufs_ext removeAllObjects];
    }
}

static struct ggml_metal_buffer_id ggml_metal_get_buffer_id(const struct ggml_tensor * t) {
    if (!t) {
        return (struct ggml_metal_buffer_id) { nil, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    return ggml_metal_buffer_get_id(buffer->context, t);
}

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    @autoreleasepool {
        // wrap the source data into a Metal buffer
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_src = [device newBufferWithBytes:data
                                                    length:size
                                                   options:MTLResourceStorageModeShared];

        GGML_ASSERT(buf_src);

        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(tensor);
        if (bid_dst.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_dst.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:buf_src
                   sourceOffset:0
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];
        [buf_src release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    @autoreleasepool {
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_dst = [device newBufferWithBytesNoCopy:data
                                                          length:size
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

        GGML_ASSERT(buf_dst);

        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(tensor);
        if (bid_src.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_src.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:buf_dst
              destinationOffset:0
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];
        [buf_dst release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    @autoreleasepool {
        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);

        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }

        // queue the copy operation into the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx_src->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:ggml_nbytes(src)];

        [encoder endEncoding];

        ggml_metal_event_t ev_cpy = ggml_metal_get_ev_cpy(ctx_src);
        ggml_metal_event_record(ctx_src, ev_cpy);

        [cmd_buf commit];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx_src->cmd_bufs_ext addObject:cmd_buf];
        ctx_src->cmd_buf_last = cmd_buf;

        [cmd_buf retain];

        ggml_metal_event_wait(ctx_dst, ev_cpy);

        return true;
    }
}

enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    // number of nodes encoded by the main thread (empirically determined)
    const int n_main = MAX(64, 0.1*gf->n_nodes);

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;

    // keep the memory wired
    ggml_metal_device_rsets_keep_alive(ctx->dev);

    // submit the ggml compute graph to the GPU by creating command buffers and encoding the ops in them
    // the first n_nodes_0 are encoded and submitted for processing directly by the calling thread
    // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
    // each thread creates it's own command buffer and enqueues the ops in parallel
    //
    // tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2

    @autoreleasepool {
        ctx->gf = gf;

        ctx->n_nodes_0 = MIN(n_main, gf->n_nodes);
        ctx->n_nodes_1 = gf->n_nodes - ctx->n_nodes_0;

        ctx->n_nodes_per_cb = (ctx->n_nodes_1 + ctx->n_cb - 1) / ctx->n_cb;

        const bool use_capture = ctx->capture_next_compute;
        if (use_capture) {
            ctx->capture_next_compute = false;

            // make sure all previous computations have finished before starting the capture
            if (ctx->cmd_buf_last) {
                [ctx->cmd_buf_last waitUntilCompleted];
                ctx->cmd_buf_last = nil;
            }

            if (!ctx->capture_started) {
                // create capture scope
                id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
                ctx->capture_scope = [[MTLCaptureManager sharedCaptureManager] newCaptureScopeWithDevice:device];

                MTLCaptureDescriptor * descriptor = [MTLCaptureDescriptor new];
                descriptor.captureObject = ctx->capture_scope;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:[NSString stringWithFormat:@"/tmp/perf-metal.gputrace"]];

                NSError * error = nil;
                if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error]) {
                    GGML_LOG_ERROR("%s: error: unable to start capture '%s'\n", __func__, [[error localizedDescription] UTF8String]);
                } else {
                    [ctx->capture_scope beginScope];
                    ctx->capture_started = true;
                }
            }
        }

        // short-hand
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);

        // the main thread commits the first few commands immediately
        // cmd_buf[n_cb]
        {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[n_cb].obj) {
                [ctx->cmd_bufs[n_cb].obj release];
            }
            ctx->cmd_bufs[n_cb].obj = cmd_buf;

            [cmd_buf enqueue];

            ctx->encode_async(n_cb);
        }

        // remember the command buffer for the next iteration
        ctx->cmd_buf_last = ctx->cmd_bufs[n_cb].obj;

        // prepare the rest of the command buffers asynchronously (optional)
        // cmd_buf[0.. n_cb)
        for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[cb_idx].obj) {
                [ctx->cmd_bufs[cb_idx].obj release];
            }
            ctx->cmd_bufs[cb_idx].obj = cmd_buf;

            // always enqueue the first two command buffers
            // enqueue all of the command buffers if we don't need to abort
            if (cb_idx < 2 || ctx->abort_callback == NULL) {
                [cmd_buf enqueue];

                // update the pointer to the last queued command buffer
                // this is needed to implement synchronize()
                ctx->cmd_buf_last = cmd_buf;
            }
        }

        dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async);

        // for debugging: block until graph is computed
        //[ctx->cmd_buf_last waitUntilCompleted];

        // enter here only when capturing in order to wait for all computation to finish
        // otherwise, we leave the graph to compute asynchronously
        if (!use_capture && ctx->capture_started) {
            // wait for completion and check status of each command buffer
            // needed to detect if the device ran out-of-memory for example (#1881)
            {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[n_cb].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, n_cb, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }
            }

            for (int i = 0; i < n_cb; ++i) {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[i].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, i, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }

                id<MTLCommandBuffer> next_buffer = (i + 1 < n_cb ? ctx->cmd_bufs[i + 1].obj : nil);
                if (!next_buffer) {
                    continue;
                }

                const bool next_queued = ([next_buffer status] != MTLCommandBufferStatusNotEnqueued);
                if (next_queued) {
                    continue;
                }

                if (ctx->abort_callback && ctx->abort_callback(ctx->abort_callback_data)) {
                    GGML_LOG_INFO("%s: command buffer %d aborted", __func__, i);
                    return GGML_STATUS_ABORTED;
                }

                [next_buffer commit];
            }

            [ctx->capture_scope endScope];
            [[MTLCaptureManager sharedCaptureManager] stopCapture];
        }
    }

    // Phase 49: GPU timing probe — print actual GPU wall-clock time per graph_compute call.
    // GPUStartTime/GPUEndTime = actual GPU execution time in seconds.
    // Gated on GGML_METAL_GPU_TIMING=1 so default behavior unchanged.
    {
        static bool gpu_timing = false;
        static dispatch_once_t gpu_timing_once;
        dispatch_once(&gpu_timing_once, ^{ gpu_timing = (getenv("GGML_METAL_GPU_TIMING") != NULL); });
        if (gpu_timing) {
            const int n_cb = ctx->n_cb;
            // Try all command buffers including index 0 (for n_cb=1 case)
            for (int ci = 0; ci <= n_cb; ci++) {
                id<MTLCommandBuffer> cb = ctx->cmd_bufs[ci].obj;
                if (cb) {
                    const double gpu_ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
                    fprintf(stderr, "*** METAL_GPU_TIME: cb%d gpu=%.2f ms status=%d\n",
                            ci, gpu_ms, (int)[cb status]);
                    fflush(stderr);
                }
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}

void ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    //const int64_t t_start = ggml_time_us();

    if (ctx->use_graph_optimize) {
        ggml_graph_optimize(gf);
    }

    //printf("%s: graph optimize took %.3f ms\n", __func__, (ggml_time_us() - t_start) / 1000.0);
}

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_signal(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_event_wait(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_wait(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

enum ggml_status ggml_metal_graph_compute_gated(
        ggml_metal_t ctx,
        struct ggml_cgraph * gf,
        int n_segments,
        const struct ggml_metal_gated_segment * segments) {
    if (!ctx || !gf || n_segments <= 0 || !segments) {
        return GGML_STATUS_FAILED;
    }

    @autoreleasepool {
        ctx->gf = gf;

        // Keep the memory residency set alive for GPU access duration.
        ggml_metal_device_rsets_keep_alive(ctx->dev);

        // 2026-04-20: honor capture_next_compute even on the gated path.
        // Without this, GGML_METAL_CAPTURE=1 wouldn't trigger for mega-
        // dispatch (which goes through this function instead of the
        // standard ggml_metal_graph_compute).
        const bool use_capture_gated = ctx->capture_next_compute;
        if (use_capture_gated) {
            ctx->capture_next_compute = false;
            if (!ctx->capture_started) {
                id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
                ctx->capture_scope = [[MTLCaptureManager sharedCaptureManager] newCaptureScopeWithDevice:device];
                MTLCaptureDescriptor * desc = [MTLCaptureDescriptor new];
                desc.captureObject = ctx->capture_scope;
                desc.destination = MTLCaptureDestinationGPUTraceDocument;
                desc.outputURL = [NSURL fileURLWithPath:@"/tmp/perf-metal.gputrace"];
                NSError * err = nil;
                if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:desc error:&err]) {
                    GGML_LOG_ERROR("%s: gated-compute capture start failed: '%s'\n",
                                   __func__, [[err localizedDescription] UTF8String]);
                } else {
                    [ctx->capture_scope beginScope];
                    ctx->capture_started = true;
                    GGML_LOG_INFO("%s: gated-compute Xcode capture started\n", __func__);
                }
            }
        }

        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);

        // SPLIT_DECODE_PER_OP_TIMING=1 — per-op GPU wall-clock timing for layer 5.
        // Encodes each layer-5 op into its own cmd_buf, commits+waits, prints GPU µs.
        // DESTROYS throughput — use only with n-predict=2 for diagnostics.
        const char * _per_op_timing_env = getenv("SPLIT_DECODE_PER_OP_TIMING");
        const BOOL per_op_timing = (_per_op_timing_env != NULL && _per_op_timing_env[0] != '\0' && _per_op_timing_env[0] != '0');

        // SPLIT_DECODE_SEGMENT_TIMING=1 — per-segment wall-clock profiling.
        // Gate behind env var so normal runs are zero-overhead.
        const char * _seg_timing_env = getenv("SPLIT_DECODE_SEGMENT_TIMING");
        const BOOL seg_timing = (_seg_timing_env != NULL && _seg_timing_env[0] != '\0' && _seg_timing_env[0] != '0');

        // SPLIT_DECODE_EVENT_WAIT_AUDIT=1 — per-layer GPU start/done times for
        // sampled layers 5, 20, 40, 60. Uses MTLCommandBuffer GPUStartTime /
        // GPUEndTime (available macOS 10.15+, measured in seconds). Printed
        // as [EVENT_WAIT_GPU] lines from the completedHandler.
        const char * _ewa_env = getenv("SPLIT_DECODE_EVENT_WAIT_AUDIT");
        const BOOL event_wait_audit = (_ewa_env != NULL && _ewa_env[0] != '\0' && _ewa_env[0] != '0');

        // SPLIT_DECODE_SINGLE_CMDBUF=1 — collapse all 78 per-segment cmd_bufs into
        // ONE mega cmd_buf. encodeWaitForEvent / encodeSignalEvent are still encoded
        // between segments so RX-thread gating still works; the Metal GPU handles
        // inter-segment waits without CPU round-trips. Per-segment timing and
        // event-wait-audit handlers are disabled in this mode (only one completedHandler
        // fires for the whole cmd_buf). PER_OP_TIMING path is unaffected.
        const char * _single_cmdbuf_env = getenv("SPLIT_DECODE_SINGLE_CMDBUF");
        const BOOL single_cmdbuf = (_single_cmdbuf_env != NULL && _single_cmdbuf_env[0] != '\0' && _single_cmdbuf_env[0] != '0')
                                   && !per_op_timing;

        if (single_cmdbuf && seg_timing) {
            GGML_LOG_WARN("%s: SPLIT_DECODE_SINGLE_CMDBUF=1 disables per-segment timing "
                          "(SPLIT_DECODE_SEGMENT_TIMING). Only one completedHandler fires.\n", __func__);
        }
        if (single_cmdbuf && event_wait_audit) {
            GGML_LOG_WARN("%s: SPLIT_DECODE_SINGLE_CMDBUF=1 disables per-segment "
                          "EVENT_WAIT_AUDIT GPU timestamps. Only end-of-dispatch times reported.\n", __func__);
        }

        // Wall-clock microseconds since an arbitrary epoch.
        // Monotonic clock, sub-microsecond resolution.
        struct timespec _seg_ts0;
        memset(&_seg_ts0, 0, sizeof(_seg_ts0));
        if (seg_timing) {
            clock_gettime(CLOCK_MONOTONIC, &_seg_ts0);
        }
        // Helper: compute wall-clock us relative to _seg_ts0. Use as inline expression.
#define SEG_NOW_US() ({ \
    struct timespec _ts_now; \
    clock_gettime(CLOCK_MONOTONIC, &_ts_now); \
    (int64_t)(_ts_now.tv_sec  - _seg_ts0.tv_sec)  * 1000000LL \
  + (int64_t)(_ts_now.tv_nsec - _seg_ts0.tv_nsec) / 1000LL; \
})

        // Per-segment timing storage (filled by addCompletedHandler callbacks).
        // Uses NSMutableDictionary per segment so the __block handler can mutate
        // the cmdbuf_done_us field after commit returns.
        NSMutableArray * seg_timing_arr = (seg_timing && !single_cmdbuf) ? [NSMutableArray arrayWithCapacity:n_segments] : nil;
        for (int s_init = 0; s_init < n_segments && seg_timing && !single_cmdbuf; ++s_init) {
            [seg_timing_arr addObject:[NSNull null]];
        }
        __block NSMutableArray * seg_timing_blk = seg_timing_arr;

        // SPLIT_DECODE_SINGLE_CMDBUF=1: create ONE cmd_buf before the loop.
        // All segments encode onto it; it is committed once after the loop.
        // This eliminates ~78 cmd_buf create/commit/retire cycles (~321 µs/layer × 78
        // = ~25 ms/token) because Metal handles inter-segment event waits GPU-side.
        id<MTLCommandBuffer> single_buf = nil;
        if (single_cmdbuf) {
            single_buf = [queue commandBufferWithUnretainedReferences];
            [single_buf retain];
        }

        // Each segment gets its own cmd_buf (default) or shares single_buf:
        //   [optional] encodeWaitForEvent(wait_event)
        //   compute encoder for nodes [start_idx, end_idx)
        //   [optional] encodeSignalEvent(signal_event)
        //   commit (per-segment in default mode; once after loop in single_cmdbuf mode)
        // Multiple cmd_bufs execute in enqueue order on the Metal queue.
        //
        // commit() implicitly enqueues — we sequence commits so execution
        // order matches our segment array order.
        for (int s = 0; s < n_segments; ++s) {
            const struct ggml_metal_gated_segment * seg = &segments[s];

            // Sanity check segment range
            if (seg->start_idx < 0 || seg->end_idx > gf->n_nodes ||
                seg->start_idx >= seg->end_idx) {
                GGML_LOG_WARN("%s: segment %d bad range [%d,%d) (n_nodes=%d)\n",
                              __func__, s, seg->start_idx, seg->end_idx, gf->n_nodes);
                continue;
            }

            // In single_cmdbuf mode reuse the one pre-created buffer.
            // In default mode allocate a fresh per-segment buffer.
            id<MTLCommandBuffer> cmd_buf;
            if (single_cmdbuf) {
                cmd_buf = single_buf;
            } else {
                cmd_buf = [queue commandBufferWithUnretainedReferences];
                [cmd_buf retain];
            }

            // Per-segment timing: record encode_start before any encoding.
            // Skipped in single_cmdbuf mode (only one completedHandler fires).
            int64_t t_encode_start = (seg_timing && !single_cmdbuf) ? SEG_NOW_US() : 0;

            // Pre-compute wait event (stall GPU until signaled).
            //
            // If caller provided an explicit wait_target, use the _value
            // variant: both the encoder and the paired host signaler derive
            // the same target from a caller-known monotonic counter (e.g.
            // token_idx + 1), eliminating the fetch_add race that the
            // original ev->value counter has when signaler and encoder run
            // on independent threads. Otherwise fall back to the plain
            // variant (used by intra-graph GPU-GPU event pairs where
            // encode_signal precedes encode_wait on the same queue).
            if (seg->wait_event) {
                if (seg->wait_target != 0) {
                    ggml_metal_event_encode_wait_value(seg->wait_event, (ggml_metal_cmd_buf_t)cmd_buf, seg->wait_target);
                } else {
                    ggml_metal_event_encode_wait(seg->wait_event, (ggml_metal_cmd_buf_t)cmd_buf);
                }
            }

            // 2026-04-21 (iter 9): fill inp_out_ids slot with 0 before compute.
            // gallocr aliases intermediate float tensors to offset 0 of the
            // compute buffer (same slot as inp_out_ids) across layers 0..N-2,
            // leaving a garbage float there by the time the last-layer segment
            // runs. Issuing a fillBuffer:range:value:0 in a blit encoder BEFORE
            // the compute encoder resets that slot to int32=0 so GET_ROWS reads
            // the correct row index. The wait event (if any) already fired above,
            // so the GPU is past the barrier and safe to issue blits.
            if (seg->fill_buf != NULL) {
                id<MTLBlitCommandEncoder> blit_enc = [cmd_buf blitCommandEncoder];
                [blit_enc fillBuffer:(id<MTLBuffer>)seg->fill_buf
                              range:NSMakeRange(seg->fill_offs, seg->fill_len)
                              value:0];
                [blit_enc endEncoding];
            }

            // SPLIT_DECODE_PER_OP_TIMING=1 — each op in the segment gets its own
            // cmd_buf with commit+waitUntilCompleted to measure actual GPU wall time.
            // Prints [PER_OP_TIME] lines for ops whose name ends in "-5" (layer 5).
            // This fully serializes all GPU work and is ONLY for diagnostics.
            if (per_op_timing) {
                // Collect all non-empty absolute node indices in this segment.
                // Mirrors the filter in ggml_metal_op's constructor.
                int abs_node_indices[2048]; // more than enough for any segment
                int n_abs = 0;
                for (int si = seg->start_idx; si < seg->end_idx && si < gf->n_nodes && n_abs < 2048; si++) {
                    struct ggml_tensor * tn = gf->nodes[si];
                    if (!ggml_op_is_empty(tn->op) && !ggml_is_empty(tn)) {
                        abs_node_indices[n_abs++] = si;
                    }
                }

                // Encode each op in its own cmd_buf. For layer-5 ops: commit+wait+print.
                // For non-layer-5 ops: commit+wait to maintain correct execution order
                // (dependencies must be satisfied).
                for (int oi = 0; oi < n_abs; oi++) {
                    int abs_ni = abs_node_indices[oi];
                    struct ggml_tensor * node = gf->nodes[abs_ni];

                    // Check if this is a layer-5 op: name ends with "-5"
                    bool is_layer5 = false;
                    if (node && node->name[0]) {
                        size_t nmlen = strlen(node->name);
                        if (nmlen >= 2 && node->name[nmlen-2] == '-' && node->name[nmlen-1] == '5') {
                            is_layer5 = true;
                        }
                    }

                    id<MTLCommandBuffer> op_buf = [queue commandBufferWithUnretainedReferences];
                    [op_buf retain];

                    ggml_metal_op_t op_ctx = ggml_metal_op_init(
                        ctx->dev,
                        (ggml_metal_cmd_buf_t)op_buf,
                        gf,
                        abs_ni,
                        abs_ni + 1,
                        /*use_fusion=*/false,
                        ctx->use_concurrency,
                        /*use_capture=*/false,
                        /*debug_graph=*/0,
                        /*debug_fusion=*/0);

                    if (ggml_metal_op_n_nodes(op_ctx) > 0) {
                        ggml_metal_op_encode(op_ctx, 0);
                    }
                    ggml_metal_op_free(op_ctx); // ends encoder

                    struct timespec ts_op_start, ts_op_end;
                    clock_gettime(CLOCK_MONOTONIC, &ts_op_start);
                    [op_buf commit];
                    [op_buf waitUntilCompleted];
                    clock_gettime(CLOCK_MONOTONIC, &ts_op_end);

                    int64_t gpu_us = (int64_t)(ts_op_end.tv_sec - ts_op_start.tv_sec) * 1000000LL
                                   + (int64_t)(ts_op_end.tv_nsec - ts_op_start.tv_nsec) / 1000LL;

                    if (is_layer5) {
                        fprintf(stderr, "[PER_OP_TIME] il=5 op=%s name=%s gpu_us=%lld\n",
                                node ? ggml_op_name(node->op) : "UNKNOWN",
                                node ? node->name : "?",
                                (long long)gpu_us);
                        fflush(stderr);
                    }

                    [ctx->cmd_bufs_ext addObject:op_buf];
                    ctx->cmd_buf_last = op_buf;
                }

                // cmd_buf (from segment loop) was never used for compute in this path.
                // Still need to commit it (empty) so segment signal_event fires on this buf.
                // We'll commit it empty below — signal_event is encoded onto it after.
            } else {
                ggml_metal_op_t ctx_op = ggml_metal_op_init(
                    ctx->dev,
                    (ggml_metal_cmd_buf_t)cmd_buf,
                    gf,
                    seg->start_idx,
                    seg->end_idx,
                    ctx->use_fusion,
                    ctx->use_concurrency,
                    /*use_capture=*/false,
                    /*debug_graph=*/0,
                    /*debug_fusion=*/0);

                for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
                    const int res = ggml_metal_op_encode(ctx_op, idx);
                    if (res == 0) {
                        break;
                    }
                    idx += res - 1;
                }

                // op_free ends the encoder via destructor
                ggml_metal_op_free(ctx_op);
            }

            // Post-compute signal event (notify host/queue that this segment's
            // work is done). The signal cmd_buf operation runs after the
            // compute commands complete on this cmd_buf. Same rationale as
            // wait_event above: if caller provided an explicit signal_target,
            // use the _value variant so the host waiter can match deterministically.
            if (seg->signal_event) {
                if (seg->signal_target != 0) {
                    ggml_metal_event_encode_signal_value(seg->signal_event, (ggml_metal_cmd_buf_t)cmd_buf, seg->signal_target);
                } else {
                    ggml_metal_event_encode_signal(seg->signal_event, (ggml_metal_cmd_buf_t)cmd_buf);
                }
            }

            // Per-segment timing: record commit time, install GPU-done handler.
            // Disabled in single_cmdbuf mode — only one completedHandler fires.
            int64_t t_commit = (seg_timing && !single_cmdbuf) ? SEG_NOW_US() : 0;
            if (seg_timing && !single_cmdbuf) {
                int seg_cap  = s;
                int64_t t_es = t_encode_start;
                int64_t t_co = t_commit;
                int32_t tok  = (seg->wait_target != 0) ? (int32_t)((int64_t)seg->wait_target - 1) : -1;
                // Store a placeholder dict; completedHandler will fill cmdbuf_done_us.
                NSMutableDictionary * d = [NSMutableDictionary dictionary];
                d[@"seg"]   = @(seg_cap);
                d[@"es"]    = @(t_es);
                d[@"co"]    = @(t_co);
                d[@"done"]  = @((int64_t)0);
                d[@"tok"]   = @(tok);
                seg_timing_blk[seg_cap] = d;
                // Capture the dict and the epoch snapshot for the handler.
                struct timespec ts0_cap = _seg_ts0;
                [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> __unused cb) {
                    struct timespec ts_done;
                    memset(&ts_done, 0, sizeof(ts_done));
                    clock_gettime(CLOCK_MONOTONIC, &ts_done);
                    int64_t done_us = (int64_t)(ts_done.tv_sec - ts0_cap.tv_sec) * 1000000LL
                                    + (int64_t)(ts_done.tv_nsec - ts0_cap.tv_nsec) / 1000LL;
                    d[@"done"] = @(done_us);
                }];
            }

            // SPLIT_DECODE_EVENT_WAIT_AUDIT: for sampled layers (seg indices 5,
            // 20, 40, 60), print [EVENT_WAIT_GPU] with GPU start/done times in µs.
            // GPUStartTime / GPUEndTime are MTLCommandBuffer properties reporting
            // GPU hardware timestamps in seconds (available macOS 10.15+).
            // Disabled in single_cmdbuf mode — per-segment GPU timestamps not
            // available when all segments share one cmd_buf.
            if (event_wait_audit && !single_cmdbuf) {
                BOOL is_sampled = (s == 5 || s == 20 || s == 40 || s == 60);
                if (is_sampled && seg->wait_target != 0) {
                    int32_t tok_ewa = (int32_t)((int64_t)seg->wait_target - 1);
                    int seg_ewa     = s;
                    [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb_ewa) {
                        // GPUStartTime and GPUEndTime are in seconds; convert to µs.
                        int64_t gpu_start_us = (int64_t)(cb_ewa.GPUStartTime * 1e6);
                        int64_t gpu_done_us  = (int64_t)(cb_ewa.GPUEndTime   * 1e6);
                        int64_t gpu_compute_us = gpu_done_us - gpu_start_us;
                        fprintf(stderr,
                            "[EVENT_WAIT_GPU] tok=%d il=%d"
                            " gpu_start_us=%lld gpu_done_us=%lld gpu_compute_us=%lld\n",
                            tok_ewa, seg_ewa,
                            (long long)gpu_start_us, (long long)gpu_done_us,
                            (long long)gpu_compute_us);
                        fflush(stderr);
                    }];
                }
            }

            // In default mode: commit this segment's cmd_buf now and track it.
            // In single_cmdbuf mode: do NOT commit per segment; we commit once
            // after the loop. Still track for cmd_bufs_ext below (added once
            // after the loop in single_cmdbuf mode).
            if (!single_cmdbuf) {
                [cmd_buf commit];
                [ctx->cmd_bufs_ext addObject:cmd_buf];
                ctx->cmd_buf_last = cmd_buf;
            }
        }

        // Single-cmdbuf mode: commit the one mega cmd_buf after all segments
        // have been encoded. Track it for the external cmd_buf list.
        if (single_cmdbuf && single_buf) {
            [single_buf commit];
            [ctx->cmd_bufs_ext addObject:single_buf];
            ctx->cmd_buf_last = single_buf;
        }

        // End the capture scope after all segments committed. The capture
        // output isn't written to disk until all cmd_bufs complete, so we
        // wait for the last one before endScope + stopCapture.
        if (use_capture_gated && ctx->capture_started) {
            if (ctx->cmd_buf_last) {
                [ctx->cmd_buf_last waitUntilCompleted];
            }
            [ctx->capture_scope endScope];
            [[MTLCaptureManager sharedCaptureManager] stopCapture];
            ctx->capture_started = false;
            GGML_LOG_INFO("%s: gated-compute Xcode capture written to /tmp/perf-metal.gputrace\n", __func__);
        }

        // SPLIT_DECODE_SEGMENT_TIMING=1: wait for all GPU completedHandlers to
        // fire, then print per-segment lines. We wait on the last cmd_buf here
        // (capture path already waited above; we only need to wait if not captured).
        if (seg_timing && seg_timing_arr.count > 0) {
            if (!use_capture_gated || !ctx->capture_started) {
                if (ctx->cmd_buf_last) {
                    [ctx->cmd_buf_last waitUntilCompleted];
                }
            }
            // Give any remaining completedHandlers a moment to fire (they fire
            // on a Metal internal queue; last-buf wait guarantees they all ran).
            int64_t total_encode = 0, total_gpu = 0;
            for (NSUInteger si = 0; si < seg_timing_arr.count; ++si) {
                id obj = seg_timing_arr[si];
                if (![obj isKindOfClass:[NSDictionary class]]) continue;
                NSDictionary * d = (NSDictionary *)obj;
                int64_t es   = [d[@"es"]   longLongValue];
                int64_t co   = [d[@"co"]   longLongValue];
                int64_t done = [d[@"done"] longLongValue];
                int     seg  = [d[@"seg"]  intValue];
                int32_t tok  = (int32_t)[d[@"tok"] intValue];
                int64_t enc_us = co - es;
                int64_t gpu_us = done - co;
                total_encode += enc_us;
                total_gpu    += (gpu_us > 0 ? gpu_us : 0);
                fprintf(stderr, "[SEG_TIME] tok=%d seg=%d encode_start=%lld commit=%lld cmdbuf_done=%lld"
                        " enc_us=%lld gpu_us=%lld\n",
                        tok, seg, (long long)es, (long long)co, (long long)done,
                        (long long)enc_us, (long long)gpu_us);
            }
            fprintf(stderr, "[SEG_TIME_TOTAL] n_segs=%d total_encode_us=%lld total_gpu_us=%lld\n",
                    n_segments, (long long)total_encode, (long long)total_gpu);
            fflush(stderr);
        }
#undef SEG_NOW_US
    }

    return GGML_STATUS_SUCCESS;
}

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx) {
    return ctx->ev_cpy;
}

ggml_metal_device_t ggml_metal_get_device(ggml_metal_t ctx) {
    return ctx->dev;
}

void ggml_metal_set_n_cb(ggml_metal_t ctx, int n_cb) {
    if (ctx->n_cb != n_cb) {
        ctx->n_cb = MIN(n_cb, GGML_METAL_MAX_COMMAND_BUFFERS);

        if (ctx->n_cb > 2) {
            GGML_LOG_WARN("%s: n_cb = %d, using n_cb > 2 is not recommended and can degrade the performance in some cases\n", __func__, n_cb);
        }
    }

    if (ctx->encode_async) {
        Block_release(ctx->encode_async);
    }

    ctx->encode_async = Block_copy(^(size_t iter) {
        const int cb_idx = iter;
        const int n_cb_l = ctx->n_cb;

        const int n_nodes_0 = ctx->n_nodes_0;
        const int n_nodes_1 = ctx->n_nodes_1;

        const int n_nodes_per_cb = ctx->n_nodes_per_cb;

        int idx_start = 0;
        int idx_end   = n_nodes_0;

        if (cb_idx < n_cb_l) {
            idx_start = n_nodes_0 + (                                         (cb_idx + 0) * n_nodes_per_cb);
            idx_end   = n_nodes_0 + (MIN((cb_idx == n_cb_l - 1) ? n_nodes_1 : (cb_idx + 1) * n_nodes_per_cb, n_nodes_1));
        }

        id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;

        ggml_metal_op_t ctx_op = ggml_metal_op_init(
            ctx->dev,
            cmd_buf,
            ctx->gf,
            idx_start,
            idx_end,
            ctx->use_fusion,
            ctx->use_concurrency,
            ctx->capture_next_compute,
            ctx->debug_graph,
            ctx->debug_fusion);

        for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
            const int res = ggml_metal_op_encode(ctx_op, idx);
            if (res == 0) {
                break;
            }

            idx += res - 1;
        }

        ggml_metal_op_free(ctx_op);

        if (cb_idx < 2 || ctx->abort_callback == NULL) {
            [cmd_buf commit];
        }
    });
}

void ggml_metal_set_abort_callback(ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data) {
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = user_data;
}

bool ggml_metal_supports_family(ggml_metal_t ctx, int family) {
    GGML_ASSERT(ctx->dev != nil);

    id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

    return [device supportsFamily:(MTLGPUFamilyApple1 + family - 1)];
}

void ggml_metal_capture_next_compute(ggml_metal_t ctx) {
    ctx->capture_next_compute = true;
}
