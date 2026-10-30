/* Round6 owner-authorized compatibility boundary. Preserve every old name.
 * Retired pure tuning controls are explicit refusal, never an inert successful set.
 * Unset controls leave upstream defaults and graph choices untouched. */
typedef struct {
    const char *name, *env;
    uint64_t group;
    int implemented, inverted, initial, maximum;
    bool explicit_value;
    int value;
} ds41_upstream_control;
static ds41_upstream_control ds41_upstream_controls[] = {
    {"decode_chunks", "DS4_DS41_DECODE_CHUNKS", 0, 1, 0, 1, 1, false, 0},
    {"decode_queue2", "DS4_DS41_DECODE_QUEUE2", 0, 1, 0, 1, 1, false, 0},
    {"narrow_weights", "DS4_DS41_NARROW_WEIGHTS", 0, 0, 0, 1, 1, false, 0},
    {"mxfp4_fixed_shapes", "DS4_DS41_MXFP4_FIXED_SHAPES", DS4_GPU_V41_FIXED_SHAPES, 1, 0, 1, 1, false, 0},
    {"queue_layers", "DS4_DS41_QUEUE_LAYERS", 0, 1, 0, 1, 1, false, 0},
    {"engram_async", "DS4_DS41_ENGRAM_ASYNC", 0, 0, 0, 1, 1, false, 0},
    {"round_fuse_norm", "DS4_METAL_DISABLE_V41_ROUND_FUSE_NORM", 0, 0, 1, 1, 1, false, 0},
    {"round_fuse_hcsum", "DS4_METAL_DISABLE_V41_ROUND_FUSE_HCSUM", 0, 0, 1, 1, 1, false, 0},
    {"round_fuse_hcexpand", "DS4_METAL_DISABLE_V41_ROUND_FUSE_HCEXPAND", 0, 0, 1, 1, 1, false, 0},
    {"q4_grouped", "DS4_METAL_ENABLE_Q4_GROUPED_EXPERTS", 0, 1, 0, 0, 1, false, 0},
    {"q4_group6", "DS4_METAL_DISABLE_Q4_GROUP6_EXPERT_TABLE", 0, 1, 1, 0, 1, false, 0},
    {"q4_group8", "DS4_METAL_ENABLE_Q4_GROUP8_EXPERT_TABLE", 0, 1, 0, 0, 1, false, 0},
    {"q4_group24", "DS4_METAL_ENABLE_Q4_GROUP24_EXPERT_TABLE", 0, 1, 0, 0, 1, false, 0},
    {"q4_wide", "DS4_DS41_Q4_WIDE", 0, 0, 0, 1, 1, false, 0},
    {"q4_gu_nr1", "DS4_DS41_Q4_GU_NR1", 0, 0, 0, 1, 1, false, 0},
    {"router_fused", "DS4_DS41_ROUTER_FUSED", 0, 0, 0, 1, 1, false, 0},
    {"router_fused_w", "DS4_DS41_ROUTER_FUSED_W", 0, 0, 0, 1, 1, false, 0},
    {"shared_swiglu", "DS4_DS41_SHARED_SWIGLU", 0, 0, 0, 1, 1, false, 0},
    {"hc_norm_mix", "DS4_DS41_HC_NORM_MIX", 0, 0, 0, 1, 1, false, 0},
    {"mtp_down_tail_pack", "DS4_DS41_MTP_DOWN_TAIL_PACK", 0, 0, 0, 1, 1, false, 0},
    {"mtp_hc_epilogue", "DS4_DS41_MTP_HC_EPILOGUE", 0, 0, 0, 1, 1, false, 0},
    {"down_tail_pack", "DS4_DS41_DOWN_TAIL_PACK", 0, 0, 0, 1, 1, false, 0},
    {"router_shared_hc", "DS4_DS41_ROUTER_SHARED_HC", 0, 0, 0, 1, 1, false, 0},
    {"q8_virtual_down", "DS4_DS41_Q8_VIRTUAL_DOWN", 0, 0, 0, 1, 1, false, 0},
    {"concurrent_down", "DS4_DS41_CONCURRENT_DOWN", 0, 0, 0, 1, 1, false, 0},
    {"attn_low_static", "DS4_DS41_ATTN_LOW_STATIC", 0, 0, 0, 1, 1, false, 0},
    {"verify_mxfp4", "DS4_DS41_VERIFY_MXFP4", 0, 0, 0, 1, 1, false, 0},
    {"mxfp4_dedup", "DS4_DS41_MXFP4_DEDUP", DS4_GPU_V41_MOE_DEDUP, 1, 0, 1, 1, false, 0},
    {"concurrent_ffn", "DS4_DS41_CONCURRENT_FFN", 0, 0, 0, 1, 1, false, 0},
    {"engram_lookup", "DS4_DS41_ENGRAM_LOOKUP", 0, 0, 0, 1, 1, false, 0},
    {"verify_output", "DS4_DS41_VERIFY_OUTPUT", 0, 0, 0, 1, 1, false, 0},
    {"rows_mma", "DS4_DS41_ROWS_MMA", DS4_GPU_V41_ROWS_MMA, 1, 0, 1, 1, false, 0},
    {"row_matmuls", "DS4_DS41_ROW_MATMULS", DS4_GPU_V41_ROW_MATMULS, 1, 0, 1, 1, false, 0},
    {"engram_gate", "DS4_DS41_ENGRAM_GATE", 0, 0, 0, 1, 1, false, 0},
    {"shared_down_hc", "DS4_DS41_SHARED_DOWN_HC", 0, 0, 0, 1, 1, false, 0},
    {"fixed_dense", "DS4_DS41_FIXED_DENSE", 0, 0, 0, 1, 1, false, 0},
    {"query_norm_pair", "DS4_DS41_QUERY_NORM_PAIR", 0, 0, 0, 1, 1, false, 0},
    {"query_rope", "DS4_DS41_QUERY_ROPE", 0, 0, 0, 1, 1, false, 0},
    {"heads_rope", "DS4_DS41_HEADS_ROPE", 0, 0, 0, 1, 1, false, 0},
    {"preencode", "DS4_DS41_PREENCODE", DS4_GPU_V41_PREENCODE, 1, 0, 1, 1, false, 0},
    {"engram_persistent", "DS4_DS41_ENGRAM_PERSISTENT", 0, 0, 0, 1, 1, false, 0},
    {"q8_virtual_query", "DS4_DS41_Q8_VIRTUAL_QUERY", 0, 0, 0, 1, 1, false, 0},
    {"kv_prepare", "DS4_DS41_KV_PREPARE", 0, 0, 0, 1, 1, false, 0},
    {"kv_query_mix", "DS4_DS41_KV_QUERY_MIX", 0, 0, 0, 1, 1, false, 0},
    {"attn_out_hc", "DS4_DS41_ATTN_OUT_HC", 0, 0, 0, 1, 1, false, 0},
    {"ffn_producer", "DS4_DS41_FFN_PRODUCER", 0, 0, 0, 1, 1, false, 0},
    {"qa_kv_flat", "DS4_DS41_QA_KV_FLAT", 0, 0, 0, 1, 1, false, 0},
    {"hc_stream_layout", "DS4_DS41_HC_STREAM_LAYOUT", 0, 0, 0, 1, 1, false, 0},
    {"hc_stream", "DS4_DS41_HC_STREAM", 0, 0, 0, 1, 1, false, 0},
    {"hc_tail", "DS4_DS41_HC_TAIL", 0, 0, 0, 1, 1, false, 0},
    {"hc_expand_fold", "DS4_DS41_HC_EXPAND_FOLD", 0, 0, 0, 1, 1, false, 0},
    {"mv_round", "DS4_DS41_MV_ROUND", 0, 0, 0, 1, 1, false, 0},
    {"producer_round", "DS4_DS41_PRODUCER_ROUND", 0, 0, 0, 1, 1, false, 0},
    {"ffn_add_fold", "DS4_DS41_FFN_ADD_FOLD", 0, 0, 0, 1, 1, false, 0},
    {"kv_stage_f16", "DS4_DS41_KV_STAGE_F16", 0, 0, 0, 1, 1, false, 0},
    {"prefill_8k_chunk", "DS4_METAL_DISABLE_V41_8K_CHUNK", 0, 1, 1, 1, 1, false, 0},
    {"prefill_decoder_suffix", "DS4_METAL_DISABLE_V41_DECODER_SUFFIX", 0, 1, 1, 1, 1, false, 0},
    {"prefill_stage_profile", "DS4_METAL_V41_STAGE_PROFILE", 0, 1, 0, 0, 1, false, 0},
    {"q8_prefill_profile", "DS4_METAL_Q8_PREFILL_PROFILE", 0, 1, 0, 0, 1, false, 0},
    {"attn_cohort4", "DS4_DS41_ATTN_COHORT4", 0, 0, 0, 1, 1, false, 0},
    {"gather_wide", "DS4_DS41_GATHER_WIDE", 0, 0, 0, 1, 1, false, 0},
    {"index_topk_radix", "DS4_DS41_INDEX_TOPK_RADIX", 0, 0, 0, 1, 1, false, 0},
    {"index_compact_score", "DS4_DS41_INDEX_COMPACT_SCORE", 0, 0, 0, 1, 1, false, 0},
    {"mtp_q8_trio_mask", "DS4_DS41_MTP_Q8_TRIO_MASK", 0, 0, 0, 1, 1, false, 0},
    {"markov_cache", "DS4_DS41_MARKOV_CACHE", 0, 0, 0, 1, 1, false, 0},
    {"markov_profile", "DS4_DS41_MARKOV_PROFILE", 0, 1, 0, 0, 1, false, 0},
    {"router_simd", "DS4_DS41_ROUTER_SIMD", 0, 0, 0, 1, 1, false, 0},
    {"router_hier", "DS4_DS41_ROUTER_HIER", 0, 0, 0, 1, 1, false, 0},
    {"mtp_down_wide", "DS4_DS41_MTP_DOWN_WIDE", 0, 0, 0, 1, 1, false, 0},
    {"mtp_gu_sort_simd", "DS4_DS41_MTP_GU_SORT_SIMD", 0, 0, 0, 1, 1, false, 0},
    {"mtp_swiglu_round", "DS4_DS41_MTP_SWIGLU_ROUND", 0, 0, 0, 1, 1, false, 0},
    {"mtp_q8_round", "DS4_DS41_MTP_Q8_ROUND", 0, 0, 0, 1, 1, false, 0},
    {"mtp_f32_rows", "DS4_DS41_MTP_F32_ROWS", 0, 0, 0, 1, 1, false, 0},
    {"mtp_pointwise_batch", "DS4_DS41_MTP_POINTWISE_BATCH", 0, 0, 0, 1, 1, false, 0},
    {"mtp_f16_trio", "DS4_DS41_MTP_F16_TRIO", 0, 0, 0, 1, 1, false, 0},
    {"mtp_gu_expert_sort", "DS4_DS41_MTP_GU_EXPERT_SORT", 0, 0, 0, 1, 1, false, 0},
    {"mtp_q8_swizzle", "DS4_DS41_MTP_Q8_SWIZZLE", 0, 0, 0, 1, 1, false, 0},
    {"mtp_gu_swizzle", "DS4_DS41_MTP_GU_SWIZZLE", 0, 0, 0, 1, 1, false, 0},
    {"index_score_stream", "DS4_DS41_INDEX_SCORE_STREAM", 0, 0, 0, 1, 1, false, 0},
    {"hc_join_static", "DS4_DS41_HC_JOIN_STATIC", 0, 0, 0, 1, 1, false, 0},
    {"topk_short", "DS4_DS41_TOPK_SHORT", 0, 0, 0, 1, 1, false, 0},
    {"index_candidates", "DS4_DS41_INDEX_CANDIDATES", 0, 0, 0, 1, 1, false, 0},
    {"index_schedule", "DS4_DS41_INDEX_SCHEDULE", 0, 0, 0, 1, 1, false, 0},
    {"index_mask_score", "DS4_DS41_INDEX_MASK_SCORE", 0, 0, 0, 1, 1, false, 0},
    {"stage_gather", "DS4_DS41_STAGE_GATHER", 0, 0, 0, 1, 1, false, 0},
    {"q4_dn_nr1", "DS4_DS41_Q4_DN_NR1", 0, 0, 0, 1, 1, false, 0},
    {"hc_expand_nth", "DS4_DS41_HC_EXPAND_NTH", 0, 0, 0, 1, 1, false, 0},
    {"routed_tail_cull", "DS4_METAL_DISABLE_V41_ROUTED_TAIL_CULL", 0, 0, 1, 1, 1, false, 0},
    {"prefill_tail_absorb", "DS4_DS41_PREFILL_TAIL_ABSORB", 0, 0, 0, 1, 1, false, 0},
    {"engram_readers", "DS4_DS41_ENGRAM_READERS", 0, 0, 0, 1, 1, false, 0},
    {"prefill_attn_rb16", "DS4_METAL_DISABLE_V41_PREFILL_ATTN_RB16", 0, 0, 1, 1, 1, false, 0},
    {"mtp_async_chunks", "DS4_DS41_MTP_ASYNC_CHUNKS", 0, 0, 0, 1, 1, false, 0},
    {"mtp_engram_rows6", "DS4_DS41_MTP_ENGRAM_ROWS6", 0, 0, 0, 1, 1, false, 0},
    {"dspark_excl_eos", "DS4_DS41_DSPARK_EXCL_EOS", 0, 1, 0, 1, 1, false, 0},
    {"mtp_capture_warmup", "DS4_DS41_MTP_CAPTURE_WARMUP", 0, 0, 0, 1, 1, false, 0},
    {"mtp_state_fix", "DS4_DS41_MTP_STATE_FIX", 0, 0, 0, 1, 1, false, 0},
    {"mtp_q8_stream6", "DS4_DS41_MTP_Q8_STREAM6", 0, 0, 0, 1, 1, false, 0},
    {"mtp_q8_stream6_mask", "DS4_DS41_MTP_Q8_STREAM6_MASK", 0, 0, 0, 1, 1, false, 0},
    {"mtp_q8_pair6", "DS4_DS41_MTP_Q8_PAIR6", 0, 0, 0, 1, 1, false, 0},
    {"prefill_f16_rows2", "DS4_DS41_PREFILL_F16_ROWS2", 0, 0, 0, 1, 1, false, 0},
    {"prefill_embed_init", "DS4_DS41_PREFILL_EMBED_INIT", 0, 0, 0, 1, 1, false, 0},
    {"prefill_hc_sum_round", "DS4_DS41_PREFILL_HC_SUM_ROUND", 0, 0, 0, 1, 1, false, 0},
    {"prefill_hc_expand_round", "DS4_DS41_PREFILL_HC_EXPAND_ROUND", 0, 0, 0, 1, 1, false, 0},
    {"prefill_hc_norm_round", "DS4_DS41_PREFILL_HC_NORM_ROUND", 0, 0, 0, 1, 1, false, 0},
    {"prefill_ffn_add_round", "DS4_DS41_PREFILL_FFN_ADD_ROUND", 0, 0, 0, 1, 1, false, 0},
    {"attn_diag", "DS4_DS41_ATTN_DIAG", 0, 0, 0, 1, 1, false, 0},
    {"prefill_attn_lean_rows", "DS4_METAL_DISABLE_V41_PREFILL_ATTN_LEAN_ROWS", 0, 0, 1, 1, 1, false, 0},
    {"prefill_engram_xsweep", "DS4_DS41_PREFILL_ENGRAM_XSWEEP", 0, 0, 0, 1, 1, false, 0},
    {"attn_geom", "DS4_DS41_ATTN_GEOM", 0, 0, 0, 1, 1, false, 0},
    {"dspark_capture", "DS4_DS41_DSPARK_CAPTURE", 0, 0, 0, 1, 1, false, 0},
    {"verify_wide_prefill", "DS4_DS41_VERIFY_WIDE_PREFILL", 0, 0, 0, 1, 1, false, 0},
    {"dspark_verify_rows", "DS4_V41_DSPARK_VERIFY_ROWS", 0, 1, 0, 0, 6, false, 0},
    {"dspark_loss_budget", "DS4_DS41_DSPARK_LOSS_BUDGET", 0, 0, 0, 1, 1, false, 0},
    {"dspark_adaptive", "DS4_DS41_DSPARK_ADAPTIVE", 0, 0, 0, 1, 1, false, 0},
    {"dspark_serve", "DS4_DS41_DSPARK_SERVE", 0, 0, 0, 1, 1, false, 0},
    {"dspark_reasoning_serial", "DS4_DS41_DSPARK_REASONING_SERIAL", 0, 0, 0, 1, 1, false, 0},
    {"dspark_expert_union", "DS4_DS41_EXPERT_UNION", 0, 1, 0, 0, 1, false, 0},
    {"verify_batch_core", "DS4_DS41_VERIFY_BATCH_CORE", 0, 0, 0, 1, 1, false, 0},
    {"dspark_force_reject", "DS4_DS41_DSPARK_FORCE_REJECT", 0, 1, 0, 0, 1, false, 0},
    {"dspark_draft_trace", "DS4_DS41_DSPARK_DRAFT_TRACE", 0, 1, 0, 0, 1, false, 0},
};
static bool ds41_upstream_controls_read;
static char ds41_upstream_controls_error[256];
static bool ds41_upstream_any_override;
static uint64_t ds41_upstream_forced_on,ds41_upstream_forced_off;
static void ds41_upstream_controls_masks(void) {
    ds41_upstream_any_override=false;
    ds41_upstream_forced_on=ds41_upstream_forced_off=0;
    for (size_t i=0;i<sizeof(ds41_upstream_controls)/sizeof(*ds41_upstream_controls);++i) {
        const ds41_upstream_control *c=&ds41_upstream_controls[i];
        if (!c->implemented || !c->explicit_value) continue;
        ds41_upstream_any_override=true;
        if(c->value) ds41_upstream_forced_on|=c->group;
        else ds41_upstream_forced_off|=c->group;
    }
}
static void ds41_upstream_controls_init(void) {
    if (ds41_upstream_controls_read) return;
    ds41_upstream_controls_read = true;
    for (size_t i=0; i<sizeof(ds41_upstream_controls)/sizeof(*ds41_upstream_controls); ++i) {
        ds41_upstream_control *c=&ds41_upstream_controls[i];
        const char *v=getenv(c->env);
        bool inverted=c->inverted;
        const char *enable=NULL,*disable=NULL;
        if(!strcmp(c->name,"q4_grouped")){enable="DS4_METAL_ENABLE_Q4_GROUPED_EXPERTS";disable="DS4_METAL_DISABLE_Q4_GROUPED_EXPERTS";}
        if(!strcmp(c->name,"q4_group6")){enable="DS4_METAL_ENABLE_Q4_GROUP6_EXPERT_TABLE";disable="DS4_METAL_DISABLE_Q4_GROUP6_EXPERT_TABLE";}
        if(!strcmp(c->name,"q4_group8")){enable="DS4_METAL_ENABLE_Q4_GROUP8_EXPERT_TABLE";disable="DS4_METAL_DISABLE_Q4_GROUP8_EXPERT_TABLE";}
        if(!strcmp(c->name,"q4_group24")){enable="DS4_METAL_ENABLE_Q4_GROUP24_EXPERT_TABLE";disable="DS4_METAL_DISABLE_Q4_GROUP24_EXPERT_TABLE";}
        if(enable && getenv(enable)) {v=getenv(enable);inverted=false;}
        if(disable && getenv(disable)) {v=getenv(disable);inverted=true;}
        if (!v) continue;
        if (!c->implemented) {
            if (!ds41_upstream_controls_error[0]) snprintf(ds41_upstream_controls_error,
                sizeof(ds41_upstream_controls_error), "retired tuning control %s (%s): Round6 upstream rebase; remove this override or use an upstream control", c->name,c->env);
            continue;
        }
        char *end=NULL; long n=strtol(v,&end,10);
        if (!*v || !end || *end || n<0 || n>c->maximum ||
            (!strcmp(c->name,"dspark_verify_rows") && n==1)) {
            if (!ds41_upstream_controls_error[0]) snprintf(ds41_upstream_controls_error,
                sizeof(ds41_upstream_controls_error), "invalid retained control %s=%s",c->env,v);
            continue;
        }
        c->value=inverted ? !n : (int)n; c->explicit_value=true;
    }
    ds41_upstream_controls_masks();
}
static ds41_upstream_control *ds41_upstream_control_find(const char *name) {
    ds41_upstream_controls_init();
    for (size_t i=0; name && i<sizeof(ds41_upstream_controls)/sizeof(*ds41_upstream_controls); ++i)
        if (!strcmp(name,ds41_upstream_controls[i].name)) return &ds41_upstream_controls[i];
    return NULL;
}
int ds41_upstream_control_value(const char *name,int upstream_default) {
    ds41_upstream_controls_init();
    if (!ds41_upstream_any_override) return upstream_default;
    ds41_upstream_control *c=ds41_upstream_control_find(name);
    if (!c || !c->implemented) {
        fprintf(stderr,"BLOCKED-OWNER: unimplemented internal control %s\n",name?name:"(null)"); abort();
    }
    return c->explicit_value ? c->value : upstream_default;
}
uint64_t ds41_upstream_control_groups(uint64_t upstream_off) {
    ds41_upstream_controls_init();
    return (upstream_off | ds41_upstream_forced_off) & ~ds41_upstream_forced_on;
}

int ds41_levers_validate_environment(char *err,size_t cap) {
    ds41_upstream_controls_init();
    if (!ds41_upstream_controls_error[0]) return 1;
    if (err && cap) snprintf(err,cap,"%s",ds41_upstream_controls_error);
    return 0;
}
int ds41_levers_require_supported(char *err,size_t cap) {
    if (!ds41_levers_validate_environment(err,cap)) return 0;
    return 1;
}
size_t ds41_levers_count(void) { return sizeof(ds41_upstream_controls)/sizeof(*ds41_upstream_controls); }
const char *ds41_levers_name(size_t i) { return i<ds41_levers_count()?ds41_upstream_controls[i].name:NULL; }
const char *ds41_levers_status(const char *name) {
    const ds41_upstream_control *c=ds41_upstream_control_find(name);
    return !c ? "unknown" : c->implemented ? "upstream_adapter" : "retired_round6";
}
int ds41_levers_get(const char *name,int *out) {
    ds41_upstream_control *c=ds41_upstream_control_find(name);
    if (!out || !c || !c->implemented) return 0;
    if (c->explicit_value) *out=c->value;
#ifdef __APPLE__
    else if (c->group) *out=(ds4_gpu_v41_off()&c->group)==0;
#endif
    else *out=c->initial;
    return 1;
}
int ds41_levers_set(const char *name,int value) {
    ds41_upstream_control *c=ds41_upstream_control_find(name);
    if (!c || !c->implemented || value<0 || value>c->maximum ||
        (!strcmp(name,"dspark_verify_rows")&&value==1)) return 0;
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    /* An encoded next-token slot must not replay the previous control state. */
    ds4_gpu_pe_drop();
#endif
    c->value=value; c->explicit_value=true;
    ds41_upstream_controls_masks();
    return 1;
}
