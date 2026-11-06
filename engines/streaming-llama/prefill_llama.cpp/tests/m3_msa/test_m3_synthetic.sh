#!/usr/bin/env bash
# MiniMax-M3 §14a block-sparse (MSA) attention COHERENCE-GATE regression test, on the tiny SYNTHETIC
# model (no real weights, no live-server disruption). Drives build3/bin/m3_logits_test.exe (which feeds
# FIXED token ids as ONE causal sequence, bypassing the dummy tokenizer) on tiny_m3_sparse.gguf.
#
# The gate (independent of the random index weight VALUES): with coverage = indexer_top_k * block_size,
#   * n_tok <= coverage  -> every block selected -> sparse logits BIT-IDENTICAL to dense  (LLAMA_M3_SPARSE 1 vs 0)
#   * n_tok >  coverage  -> real block pruning   -> sparse logits DIVERGE from dense
# A clean pass at the EXACT boundary proves the MSA block-selection + masked attention end-to-end.
#
# Regenerate the gguf:  python scripts/make_tiny_m3_gguf.py FIXTURE_DIRECTORY
# Build the harness:    cl /std:c++17 /EHsc /I include /I ggml/include tests\m3_msa\m3_logits_test.cpp \
#                          /Fe:build3\bin\m3_logits_test.exe /link build3\src\llama.lib
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
EXE="${EXE:-$ROOT/build3/bin/m3_logits_test.exe}"
M="${M:-/d/tmp/tiny_m3/tiny_m3_sparse.gguf}"
COVERAGE="${COVERAGE:-8}"   # indexer_top_k(2) * sparse_block_size(4) for the tiny config
export PATH="$ROOT/build3/bin:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin:$PATH"

if [ ! -x "$EXE" ]; then echo "SKIP: $EXE not built"; exit 0; fi
if [ ! -f "$M" ]; then echo "SKIP: $M missing (run scripts/make_tiny_m3_gguf.py)"; exit 0; fi

# CPU-only (CUDA_VISIBLE_DEVICES=-1): the sparse-graph LOGIC is backend-independent, and CPU is
# deterministic. (On CUDA the FA-off attention path has a cuda-graph capture issue, and per-group's
# per-head mask needs FA off — both orthogonal real-model concerns, not the selection logic.)
ck() { CUDA_VISIBLE_DEVICES=-1 LLAMA_M3_SPARSE_PERGROUP="$PG" LLAMA_M3_SPARSE="$1" "$EXE" "$M" "$2" 2>/dev/null | grep -o "argmax=[0-9]* max=[0-9.]* l0=[-0-9.]*"; }

run_gate() {
  local label="$1"; local lfail=0
  echo "M3 MSA coherence gate [$label] (coverage = topk*block_size = $COVERAGE):"
  for N in 4 8 9 12 16; do
    d="$(ck 0 "$N")"; s="$(ck 1 "$N")"
    if [ -z "$s" ]; then echo "  n_tok=$N  FAIL: sparse produced no output (crash)"; lfail=1; continue; fi
    if [ "$N" -le "$COVERAGE" ]; then
      if [ "$d" = "$s" ]; then echo "  n_tok=$N  PASS full-coverage == dense"; else echo "  n_tok=$N  FAIL full-coverage should == dense: d{$d} s{$s}"; lfail=1; fi
    else
      if [ "$d" != "$s" ]; then echo "  n_tok=$N  PASS partial-coverage diverges (real pruning)"; else echo "  n_tok=$N  FAIL partial-coverage should DIVERGE: $d"; lfail=1; fi
    fi
  done
  return $lfail
}

# DECODE-path consistency: the LAST token's argmax must be the SAME computed one-shot (prefill-of-N) vs
# incrementally (prefill-(N-1) + decode-1). This validates the indexer-K cache store/view + decode scoring
# (the coherence gate can't — it's score-independent). argmax-only: max has ~1e-6 f32 non-associativity.
am() { CUDA_VISIBLE_DEVICES=-1 LLAMA_M3_TEST_INCR="$3" LLAMA_M3_SPARSE_PERGROUP="$PG" LLAMA_M3_SPARSE=1 \
       "$EXE" "$M" "$1" 2>/dev/null | grep -o "argmax=[0-9]*"; }
run_decode_consistency() {
  local label="$1"; local lfail=0
  echo "M3 MSA decode-consistency [$label] (one-shot vs prefill+decode, argmax must match):"
  for N in 6 12 16; do
    o="$(am "$N" x "")"; i="$(am "$N" x 1)"
    if [ -n "$o" ] && [ "$o" = "$i" ]; then echo "  n_tok=$N  PASS oneshot==incremental ($o)"; else echo "  n_tok=$N  FAIL oneshot{$o} != incremental{$i}"; lfail=1; fi
  done
  return $lfail
}

# INIT/LOCAL-block FORCE: a force that covers all causally-valid blocks must make sparse == dense even
# at topk=1 (the sink/recent blocks are selected regardless of score); WITHOUT the force, topk=1 prunes.
amf() { CUDA_VISIBLE_DEVICES=-1 LLAMA_M3_TOPK="$2" LLAMA_M3_INIT_BLOCKS="$3" LLAMA_M3_LOCAL_BLOCKS="$4" \
        LLAMA_M3_SPARSE="$5" "$EXE" "$M" "$1" 2>/dev/null | grep -o "argmax=[0-9]*"; }
run_force() {
  local lfail=0
  echo "M3 MSA init/local-block force (topk=1; cover-all force => ==dense; no force => prune):"
  for N in 12 16; do
    de="$(amf "$N" 1 0 0 0)"                 # dense
    loc="$(amf "$N" 1 0 64 1)"; ini="$(amf "$N" 1 64 0 1)"; off="$(amf "$N" 1 0 0 1)"
    if [ "$de" = "$loc" ]; then echo "  n_tok=$N  PASS local-force==dense ($loc)"; else echo "  n_tok=$N  FAIL local-force{$loc} != dense{$de}"; lfail=1; fi
    if [ "$de" = "$ini" ]; then echo "  n_tok=$N  PASS init-force==dense ($ini)";  else echo "  n_tok=$N  FAIL init-force{$ini} != dense{$de}";  lfail=1; fi
    if [ "$de" != "$off" ]; then echo "  n_tok=$N  PASS topk=1-no-force diverges ($off)"; else echo "  n_tok=$N  FAIL topk=1-no-force should prune"; lfail=1; fi
  done
  return $lfail
}

# FA-ON production path (shared mask): get_kq_mask is F16 with FA on; the graph casts it to F32 before
# the causal add. Runs on the DEFAULT device (CUDA if present — the real path; else CPU). Shared mask
# only (FA can't do the per-head mask). Full coverage must == dense, partial must diverge.
faon() { LLAMA_M3_TEST_FA=1 LLAMA_M3_SPARSE="$2" "$EXE" "$M" "$1" 2>/dev/null | grep -o "argmax=[0-9]*"; }
run_fa_production() {
  local lfail=0
  echo "M3 MSA FA-ON production path (shared mask, default device):"
  for N in 8 12; do
    d="$(faon "$N" 0)"; s="$(faon "$N" 1)"
    if [ -z "$s" ]; then echo "  n_tok=$N  FAIL: FA-on sparse no output"; lfail=1; continue; fi
    if [ "$N" -le "$COVERAGE" ]; then
      [ "$d" = "$s" ] && echo "  n_tok=$N  PASS full-coverage == dense ($s)" || { echo "  n_tok=$N  FAIL FA-on full should ==dense: d{$d} s{$s}"; lfail=1; }
    else
      [ "$d" != "$s" ] && echo "  n_tok=$N  PASS partial diverges ($s)" || { echo "  n_tok=$N  FAIL FA-on partial should prune"; lfail=1; }
    fi
  done
  return $lfail
}

fail=0
run_fa_production || fail=1
PG=0 run_gate "shared mask (default)" || fail=1
PG=1 run_gate "per-group masks (LLAMA_M3_SPARSE_PERGROUP=1)" || fail=1
PG=0 run_decode_consistency "shared mask" || fail=1
PG=1 run_decode_consistency "per-group masks" || fail=1
PG=0 run_force || fail=1
if [ "$fail" = 0 ]; then echo "ALL PASS"; else echo "FAILED"; fi
exit $fail
