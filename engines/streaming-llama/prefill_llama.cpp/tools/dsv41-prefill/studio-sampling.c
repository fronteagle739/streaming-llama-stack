/* Derived from vendored ds4.c; original MIT license retained at engines/ds4-win/LICENSE.
 * Source SHA-256: d25b29076ab04530217d41a266256b0890dca8e97db98bea15aa3614ee0f1cc7. Refresh with scripts/export_ds4_sampling.py.
 * Shared Studio/native sampler: temperature, top-k, top-p, min-p and seeded RNG.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#define DS4_NEG_INF (-INFINITY)
static void *xmalloc(size_t n) { void *p=malloc(n); if(!p) abort(); return p; }
static int sample_argmax(const float *logits, uint32_t n) {
    int best=0; float v=DS4_NEG_INF;
    for(uint32_t i=0;i<n;i++) if(logits[i]>v) {v=logits[i];best=(int)i;}
    return best;
}
typedef struct {int id;float logit;float prob;} sample_candidate;
static uint64_t sample_rng_next(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0) x = 0x9e3779b97f4a7c15ULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545f4914f6cdd1dULL;
}

static float sample_rng_f32(uint64_t *state) {
    const uint64_t x = sample_rng_next(state);
    return (float)((x >> 40) & 0xffffffu) / 16777216.0f;
}

static int sample_candidate_cmp_desc(const void *a, const void *b) {
    const sample_candidate *ca = a;
    const sample_candidate *cb = b;
    const int logit_order =
        (cb->logit > ca->logit) - (cb->logit < ca->logit);
    if (logit_order != 0) return logit_order;
    return (ca->id > cb->id) - (ca->id < cb->id);
}

static bool sample_candidate_gt(sample_candidate a, sample_candidate b) {
    if (a.logit != b.logit) return a.logit > b.logit;
    return a.id < b.id;
}

static void sample_heap_sift_up(sample_candidate *heap, uint32_t idx) {
    while (idx > 0) {
        const uint32_t parent = (idx - 1u) / 2u;
        if (!sample_candidate_gt(heap[parent], heap[idx])) break;
        sample_candidate tmp = heap[parent];
        heap[parent] = heap[idx];
        heap[idx] = tmp;
        idx = parent;
    }
}

static void sample_heap_sift_down(sample_candidate *heap, uint32_t n, uint32_t idx) {
    for (;;) {
        const uint32_t left = idx * 2u + 1u;
        const uint32_t right = left + 1u;
        uint32_t smallest = idx;
        if (left < n && sample_candidate_gt(heap[smallest], heap[left])) {
            smallest = left;
        }
        if (right < n && sample_candidate_gt(heap[smallest], heap[right])) {
            smallest = right;
        }
        if (smallest == idx) break;
        sample_candidate tmp = heap[idx];
        heap[idx] = heap[smallest];
        heap[smallest] = tmp;
        idx = smallest;
    }
}

static bool sample_fast_top_p(
        const float *logits,
        uint32_t     n_vocab,
        uint32_t     finite,
        float        max_logit,
        int          best,
        float        temperature,
        float        top_p,
        float        min_p,
        uint64_t    *rng,
        int         *token_out) {
    enum { SAMPLE_FAST_TOP_P_CAP = 512 };
    if (!logits || !rng || !token_out || finite == 0) return false;
    if (finite > SAMPLE_FAST_TOP_P_CAP && top_p >= 0.999f) return false;

    const uint32_t cap = finite < SAMPLE_FAST_TOP_P_CAP ?
        finite : (uint32_t)SAMPLE_FAST_TOP_P_CAP;
    sample_candidate heap[SAMPLE_FAST_TOP_P_CAP];
    uint32_t n = 0;
    float sum = 0.0f;
    float heap_sum = 0.0f;

    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        const float p = expf((v - max_logit) / temperature);
        sum += p;
        sample_candidate cand = {.id = (int)i, .logit = v, .prob = p};
        if (n < cap) {
            heap[n] = cand;
            heap_sum += p;
            sample_heap_sift_up(heap, n);
            n++;
        } else if (sample_candidate_gt(cand, heap[0])) {
            heap_sum -= heap[0].prob;
            heap[0] = cand;
            heap_sum += p;
            sample_heap_sift_down(heap, n, 0);
        }
    }
    if (sum <= 0.0f || !isfinite(sum)) {
        *token_out = best;
        return true;
    }

    if (n < finite && heap_sum < top_p * sum) {
        return false;
    }

    qsort(heap, n, sizeof(heap[0]), sample_candidate_cmp_desc);
    const float min_prob = (heap[0].prob / sum) * (min_p > 0.0f ? min_p : 0.0f);
    const float min_prob_raw = heap[0].prob * (min_p > 0.0f ? min_p : 0.0f);
    float filtered_sum = 0.0f;
    uint32_t filtered = 0;
    bool stopped_by_min_p = false;
    for (uint32_t i = 0; i < n; i++) {
        const float p = heap[i].prob / sum;
        if (i > 0 && p < min_prob) {
            stopped_by_min_p = true;
            break;
        }
        filtered_sum += heap[i].prob;
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (n < finite &&
        stopped_by_min_p &&
        min_p > 0.0f &&
        heap[n - 1u].prob >= min_prob_raw) {
        return false;
    }
    if (filtered == 0) {
        *token_out = best;
        return true;
    }

    float r = sample_rng_f32(rng) * filtered_sum;
    for (uint32_t i = 0; i < filtered; i++) {
        r -= heap[i].prob;
        if (r <= 0.0f) {
            *token_out = heap[i].id;
            return true;
        }
    }
    *token_out = heap[filtered - 1u].id;
    return true;
}

static int sample_full_vocab(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        float        top_p,
        float        min_p,
        uint64_t    *rng,
        float       *prob_scratch) {
    float max_logit = DS4_NEG_INF;
    int best = 0;
    uint32_t finite = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        finite++;
        if (v > max_logit) {
            max_logit = v;
            best = (int)i;
        }
    }
    if (finite == 0) return sample_argmax(logits, n_vocab);

    int fast_token = best;
    if (top_p < 1.0f &&
        sample_fast_top_p(logits,
                          n_vocab,
                          finite,
                          max_logit,
                          best,
                          temperature,
                          top_p,
                          min_p,
                          rng,
                          &fast_token)) {
        return fast_token;
    }

    if (top_p >= 1.0f) {
        float sum = 0.0f;
        const float min_rel = min_p > 0.0f ? min_p : 0.0f;
        if (min_rel > 1.0f) return best;

        /* Find a conservative log-space rejection boundary using the same
         * expf implementation as the probability path. Values below this
         * boundary are guaranteed to fail min-p, avoiding an expf for the
         * overwhelming majority of a large vocabulary. Near-boundary values
         * still take the ordinary expf comparison. */
        float reject_scaled = DS4_NEG_INF;
        bool have_reject_scaled = false;
        if (min_rel > 0.0f && isfinite(min_rel)) {
            float cutoff = logf(min_rel);
            for (int i = 0; i < 8 && isfinite(cutoff); i++) {
                cutoff = nextafterf(cutoff, -FLT_MAX);
                if (expf(cutoff) < min_rel) {
                    reject_scaled = cutoff;
                    have_reject_scaled = true;
                    break;
                }
            }
        }

        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            prob_scratch[i] = -1.0f;
            if (!isfinite(v)) continue;
            const float scaled = (v - max_logit) / temperature;
            if (have_reject_scaled && scaled <= reject_scaled) continue;
            const float p = expf(scaled);
            if (p < min_rel) continue;
            prob_scratch[i] = p;
            sum += p;
        }
        if (sum <= 0.0f || !isfinite(sum)) return best;
        float r = sample_rng_f32(rng) * sum;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float p = prob_scratch[i];
            if (p < 0.0f) continue;
            r -= p;
            if (r <= 0.0f) return (int)i;
        }
        return best;
    }

    uint32_t n = 0;
    float sum = 0.0f;
    sample_candidate *cand = NULL;
    if (min_p > 0.0f && min_p <= 1.0f) {
        /* The later min-p comparison is equivalent to
         * exp((logit-max)/temperature) >= min_p; its normalization cancels.
         * Still compute the full softmax sum in the original order, then sort
         * only candidates that can survive. This preserves the nucleus mass
         * and RNG semantics while avoiding a full-vocabulary qsort. */
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            prob_scratch[i] = -1.0f;
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            prob_scratch[i] = p;
            sum += p;
        }
        if (sum <= 0.0f || !isfinite(sum)) return best;

        const float min_prob = (1.0f / sum) * min_p;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float p = prob_scratch[i];
            if (p < 0.0f || p / sum < min_prob) continue;
            n++;
        }
        if (n == 0) return best;
        cand = xmalloc((size_t)n * sizeof(cand[0]));
        uint32_t out = 0;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float p = prob_scratch[i];
            if (p < 0.0f || p / sum < min_prob) continue;
            cand[out++] = (sample_candidate){
                .id = (int)i, .logit = logits[i], .prob = p
            };
        }
    } else {
        cand = xmalloc((size_t)finite * sizeof(cand[0]));
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            cand[n++] = (sample_candidate){.id = (int)i, .logit = v, .prob = p};
            sum += p;
        }
    }
    if (sum <= 0.0f || !isfinite(sum)) {
        free(cand);
        return best;
    }

    qsort(cand, n, sizeof(cand[0]), sample_candidate_cmp_desc);
    const float min_prob = (cand[0].prob / sum) * (min_p > 0.0f ? min_p : 0.0f);
    float filtered_sum = 0.0f;
    uint32_t filtered = 0;
    for (uint32_t i = 0; i < n; i++) {
        const float p = cand[i].prob / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += cand[i].prob;
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered == 0) {
        free(cand);
        return best;
    }

    float r = sample_rng_f32(rng) * filtered_sum;
    for (uint32_t i = 0; i < filtered; i++) {
        r -= cand[i].prob;
        if (r <= 0.0f) {
            const int id = cand[i].id;
            free(cand);
            return id;
        }
    }
    const int id = cand[filtered - 1].id;
    free(cand);
    return id;
}

static int sample_top_p_min_p(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        int          top_k,
        float        top_p,
        float        min_p,
        uint64_t    *rng,
        float       *prob_scratch) {
    if (temperature <= 0.0f) return sample_argmax(logits, n_vocab);
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;
    if (top_k <= 0) {
        const bool owned_scratch = prob_scratch == NULL;
        if (owned_scratch) {
            prob_scratch = xmalloc((size_t)n_vocab * sizeof(prob_scratch[0]));
        }
        const int token = sample_full_vocab(logits, n_vocab, temperature,
                                            top_p, min_p, rng, prob_scratch);
        if (owned_scratch) free(prob_scratch);
        return token;
    }
    if (top_k > 1024) top_k = 1024;
    if ((uint32_t)top_k > n_vocab) top_k = (int)n_vocab;

    int ids[1024];
    float vals[1024];
    int n = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        float v = logits[i];
        if (!isfinite(v)) continue;
        if (n == top_k && v <= vals[n - 1]) continue;
        int j = n < top_k ? n++ : n - 1;
        while (j > 0 && vals[j - 1] < v) {
            vals[j] = vals[j - 1];
            ids[j] = ids[j - 1];
            j--;
        }
        vals[j] = v;
        ids[j] = (int)i;
    }
    if (n == 0) return sample_argmax(logits, n_vocab);

    float probs[1024];
    const float max_logit = vals[0];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf((vals[i] - max_logit) / temperature);
        sum += probs[i];
    }
    if (sum <= 0.0f || !isfinite(sum)) return ids[0];

    const float min_prob = (probs[0] / sum) * min_p;
    float filtered_sum = 0.0f;
    int filtered = 0;
    for (int i = 0; i < n; i++) {
        float p = probs[i] / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += probs[i];
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered <= 0) return ids[0];

    float r = sample_rng_f32(rng) * filtered_sum;
    for (int i = 0; i < filtered; i++) {
        r -= probs[i];
        if (r <= 0.0f) return ids[i];
    }
    return ids[filtered - 1];
}
int slstack_sample_logits(const float *logits, int n, float temperature,
                         int top_k, float top_p, float min_p, uint64_t *rng) {
    return sample_top_p_min_p(logits, (uint32_t)n, temperature, top_k, top_p, min_p, rng, NULL);
}
