/* Compatibility observations over the upstream graph and admission policy.
 * This file is included after the public speculative APIs. No second decoder. */
#ifdef DS4_HAS_DEEPSEEK41_GPU
static int ds41_telemetry_argmax(const ds41_dspark *d, const float *logits) {
    if (!d || !d->telemetry_exclude_eos) return sample_argmax(logits, DS4_N_VOCAB);
    int best = -1;
    for (uint32_t i = 0; i < DS4_N_VOCAB; ++i)
        if ((int)i != d->telemetry_eos && (best < 0 || logits[i] > logits[best])) best = (int)i;
    return best;
}

static bool ds41_telemetry_expert_union(ds41_gpu_graph *g, uint32_t count) {
    ds41_dspark *d = g->dspark;
    if (!d || !d->telemetry_enabled || !d->telemetry_union || !g->verify_prefix) return true;
    if (!count || count > DS4_TP_BATCH_MAX_ROWS || DS4_N_EXPERT_USED > 16 || DS4_N_EXPERT > 1024)
        return false;
    uint32_t selected[DS4_TP_BATCH_MAX_ROWS * 16]; unsigned char seen[1024] = {0};
    if (!ds4_gpu_end_commands()) return false;
    bool ok = ds4_gpu_tensor_read(g->batch.selected, 0, selected,
                                 (uint64_t)count * DS4_N_EXPERT_USED * sizeof(uint32_t));
    uint32_t n = 0;
    for (uint32_t i = 0; ok && i < count * DS4_N_EXPERT_USED; ++i) {
        if (selected[i] >= DS4_N_EXPERT) { ok = false; break; }
        if (!seen[selected[i]]) { seen[selected[i]] = 1; ++n; }
    }
    if (ok) {
        d->telemetry_union_sum += n; ++d->telemetry.union_layers;
        d->telemetry.expert_union = (double)d->telemetry_union_sum / d->telemetry.union_layers;
    }
    return ds4_gpu_begin_commands() && ok;
}
/* A failed draft never mutates the target. Once verification starts, recovery
 * requires all saved owners and a successful GPU drain; otherwise poison it. */
static int ds41_telemetry_fault_recover(ds4_session *s, int first_token, uint32_t start,
    bool verified, uint64_t *rng, uint64_t saved_rng, int *accepted, char *err, size_t errlen) {
    ds41_gpu_graph *g = &s->ds41_graph;
    ds41_dspark *d = g->dspark;
    bool drained = d->fault_gpu_drained;
    if (ds4_gpu_commands_active() && !ds4_gpu_end_commands()) drained = false;
    drained = drained && !s->engine->tp.active && g->pos == start &&
        s->checkpoint_valid && (uint32_t)s->checkpoint.len == start;
    if (drained && verified) {
        drained = d->fault_saved_rows >= 2 && d->verify_start == start;
        if (drained) {
            d->verify_count = d->fault_saved_rows; d->rewind_end = 0;
            memcpy(d->host_verified + (uint64_t)DS41_VERIFY_ROWS * DS4_N_VOCAB,
                   s->logits, (size_t)DS4_N_VOCAB * sizeof(float));
            drained = ds41_dspark_commit(d, g, 0, s->logits);
            if (drained) g->valid = true;
        }
    }
    ds41_fault_latch(&s->ds41_dspark_fault, drained);
    fprintf(stderr, "ds4: dspark fault event=latched request=%llu attempts=%llu failures=%llu skips=%llu disabled=1 safe=%d\n",
        (unsigned long long)s->ds41_dspark_fault.requests,
        (unsigned long long)s->ds41_dspark_fault.attempts,
        (unsigned long long)s->ds41_dspark_fault.failures,
        (unsigned long long)s->ds41_dspark_fault.skips, drained);
    if (ds41_fault_skip(&s->ds41_dspark_fault)) {
        if (rng) *rng = saved_rng;
        if (ds4_session_eval(s, first_token, err, errlen)) return -1;
        if (d->telemetry_enabled) {
            d->telemetry.serial_rows++; d->telemetry.committed++;
            d->telemetry.controller_serial++;
        }
        accepted[0] = first_token;
        return 1;
    }
    g->valid = s->checkpoint_valid = false;
    snprintf(err, errlen, "V4.1 DSpark failure could not restore its frontier at %u", start);
    if (s->engine->tp.active) ds4_tp_mark_failed(s->engine->tp.ctx);
    return -1;
}

#endif

void ds4_session_ds41_dspark_request_begin(ds4_session *s) {
#ifdef DS4_HAS_DEEPSEEK41_GPU
    if (!s || !ds4_session_is_ds41(s)) return;
    ds41_fault_begin_request(&s->ds41_dspark_fault);
    if (!s->ds41_graph.dspark) return;
    ds41_dspark *d = s->ds41_graph.dspark;
    d->telemetry_prior_cycles += d->telemetry.cycles;
    d->verify_artifact_failed = false;
    d->telemetry_prior_committed += d->telemetry.committed;
    memset(&d->telemetry, 0, sizeof(d->telemetry));
    d->telemetry_enabled = true; d->telemetry_exclude_eos = false;
    d->telemetry_union_sum = 0;
    int enabled = 0;
    d->telemetry_union = ds41_levers_get("dspark_expert_union", &enabled) && enabled;
    ds41_admit_begin(&d->admission); d->admission_end = s->ds41_graph.pos;
#else
    (void)s;
#endif
}

int ds4_session_ds41_dspark_usage(const ds4_session *s, uint64_t out[5]) {
    if (out) memset(out, 0, 5 * sizeof(*out));
#ifdef DS4_HAS_DEEPSEEK41_GPU
    if (!out || !s || !ds4_session_is_ds41(s) || !s->ds41_graph.dspark ||
        !s->ds41_graph.dspark->telemetry_enabled) return 0;
    const ds4_dspark_decode_stats *t = &s->ds41_graph.dspark->telemetry;
    out[0] = s->ds41_graph.dspark->telemetry_prior_cycles + t->cycles;
    out[1] = s->ds41_graph.dspark->telemetry_prior_committed + t->committed; out[2] = t->controller_serial;
    out[3] = t->controller_attempts; out[4] = t->controller_declines;
    return 1;
#else
    (void)s; return 0;
#endif
}

int ds4_session_ds41_dspark_adaptive_stats(const ds4_session *s, uint64_t out[7], double ms[3]) {
    if (out) memset(out, 0, 7 * sizeof(*out)); if (ms) memset(ms, 0, 3 * sizeof(*ms));
#ifdef DS4_HAS_DEEPSEEK41_GPU
    if (!s || !ds4_session_is_ds41(s) || !s->ds41_graph.dspark ||
        !s->ds41_graph.dspark->telemetry_enabled) return 0;
    const ds4_dspark_decode_stats *t = &s->ds41_graph.dspark->telemetry;
    if (out) {
        out[0]=t->controller_attempts; out[1]=t->controller_serial; out[2]=t->controller_declines;
        out[3]=t->controller_windows; out[4]=t->controller_backoffs; out[5]=t->controller_losing;
        out[6]=t->controller_confidence_declines;
    }
    if (ms) { ms[0]=t->controller_serial_ms; ms[1]=t->controller_cycle_ms; ms[2]=t->controller_paid_ms; }
    return 1;
#else
    (void)s; return 0;
#endif
}

int ds4_session_ds41_dspark_fault_stats(const ds4_session *s, uint64_t out[4]) {
    if (out) memset(out, 0, 4 * sizeof(*out));
#ifdef DS4_HAS_DEEPSEEK41_GPU
    if (!s || !ds4_session_is_ds41(s)) return 0;
    const ds41_dspark_fault *f = &s->ds41_dspark_fault;
    if (out) {
        out[0]=f->requests; out[1]=f->attempts;
        out[2]=f->failures; out[3]=f->skips;
    }
    return f->disabled ? 2 : 1;
#else
    (void)s; return 0;
#endif
}

int ds4_session_dspark_verify_artifact_failed(const ds4_session *s) {
#ifdef DS4_HAS_DEEPSEEK41_GPU
    return s && ds4_session_is_ds41(s) && s->ds41_graph.dspark &&
        s->ds41_graph.dspark->verify_artifact_failed;
#else
    (void)s; return 0;
#endif
}

int ds4_session_dspark_decode_stats_snapshot(const ds4_session *s, ds4_dspark_decode_stats *out) {
#ifdef DS4_HAS_DEEPSEEK41_GPU
    if (!out || !s || !ds4_session_is_ds41(s) || !s->ds41_graph.dspark ||
        !s->ds41_graph.dspark->telemetry_enabled) return 0;
    *out = s->ds41_graph.dspark->telemetry;
    return 1;
#else
    (void)s; (void)out; return 0;
#endif
}

int ds4_session_dspark_generate(ds4_session *s, int gen_tokens, int eos_id,
                                int *out_tokens, int *n_out, ds4_dspark_decode_stats *st,
                                char *err, size_t errlen) {
    if (n_out) *n_out = 0; if (st) memset(st, 0, sizeof(*st));
#if defined(__APPLE__) && defined(DS4_HAS_DEEPSEEK41_GPU)
    if (!s || !n_out || !out_tokens || gen_tokens <= 0 || !ds4_session_is_ds41(s) ||
        !s->ds41_graph.dspark || !s->checkpoint_valid || !s->ds41_graph.valid) {
        payload_set_err(err, errlen, "DSpark benchmark requires a complete Metal V4.1 drafter frontier"); return 1;
    }
    ds41_dspark *d = s->ds41_graph.dspark;
    ds4_session_ds41_dspark_request_begin(s);
    int exclude = 1; (void)ds41_levers_get("dspark_excl_eos", &exclude);
    d->telemetry_exclude_eos = exclude != 0; d->telemetry_eos = eos_id;
    int token = ds4_session_argmax_excluding(s, exclude ? eos_id : -1);
    if (token < 0) { d->telemetry_exclude_eos = false;
        payload_set_err(err, errlen, "DSpark benchmark frontier has no token"); return 1; }
    out_tokens[(*n_out)++] = token;
    const double started = now_sec(); int rc = 0;
    /* Preserve the retained benchmark protocol: emit the current frontier,
     * evaluate N-1 tokens, and leave the final emitted token unevaluated. */
    while (*n_out < gen_tokens && (exclude || token != eos_id)) {
        if (ds4_session_cancelled(s)) { rc = DS4_SESSION_SYNC_INTERRUPTED; break; }
        int accepted[DS41_VERIFY_ROWS];
        int n = ds4_session_eval_speculative_argmax(s, token, gen_tokens - *n_out,
                                                    exclude ? -1 : eos_id, accepted, DS41_VERIFY_ROWS, err, errlen);
        if (n <= 0) { rc=1; break; }
        for (int i = 1; i < n; ++i) out_tokens[(*n_out)++] = accepted[i];
        if (!exclude && accepted[n-1] == eos_id) break;
        token = ds4_session_argmax_excluding(s, exclude ? eos_id : -1);
        if (token < 0) { payload_set_err(err, errlen, "DSpark benchmark frontier has no token"); rc=1; break; }
        out_tokens[(*n_out)++] = token;
    }
    d->telemetry.total_ms = (now_sec() - started) * 1000;
    if (st) *st = d->telemetry;
    d->telemetry_exclude_eos = false;
    return rc;
#else
    (void)s; (void)gen_tokens; (void)eos_id; (void)out_tokens;
    payload_set_err(err, errlen, "DSpark benchmark requires Metal V4.1"); return 1;
#endif
}
