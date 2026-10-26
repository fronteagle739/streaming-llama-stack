"""Refresh the narrow native sampler export from the vendored ds4 implementation.

Retains ds4's RNG, filtering, ordering and probability arithmetic. Only the
argmax implementation is scalar here (same stable lowest-ID tie break).
"""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'engines/ds4-win/ds4.c'
text = source.read_text(encoding='utf-8')
def function(name):
    import re
    match = re.search(r'^static [^\n]+\b' + name + r'\(', text, re.M)
    if not match:raise ValueError(name)
    start = match.start();end = text.index('\n}', match.end()) + 2
    return text[start:end]

names = ['sample_rng_next', 'sample_rng_f32', 'sample_candidate_cmp_desc',
         'sample_candidate_gt', 'sample_heap_sift_up', 'sample_heap_sift_down',
         'sample_fast_top_p', 'sample_full_vocab', 'sample_top_p_min_p']
out = '''/* Derived from vendored ds4.c; original MIT license retained at engines/ds4-win/LICENSE.
 * Source SHA-256: SOURCE_HASH. Refresh with scripts/export_ds4_sampling.py.
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
'''.replace('SOURCE_HASH', hashlib.sha256(source.read_bytes()).hexdigest())
out += '\n\n'.join(function(name) for name in names)
out += '''
int slstack_sample_logits(const float *logits, int n, float temperature,
                         int top_k, float top_p, float min_p, uint64_t *rng) {
    return sample_top_p_min_p(logits, (uint32_t)n, temperature, top_k, top_p, min_p, rng, NULL);
}
'''
target = ROOT / 'engines/streaming-llama/prefill_llama.cpp/tools/dsv41-prefill/studio-sampling.c'
target.write_text(out, encoding='utf-8', newline='\n')
print(target)
