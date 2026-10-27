#!/bin/sh
# Fork-vs-upstream acceptance for a ported arch: tiny random-weight GGUFs, logits over a 64-token prompt
# plus 8 single-token decode steps (72 rows), single ubatch (64) and multi-ubatch (16), fork vs upstream.
# Flash attention on: compared fork vs upstream at the same ubatch size only. With a sparse indexer a near-tie
# top-k score can pick a different pool under a different summation order, and upstream's own FA run is not
# ubatch-invariant on these fixtures (on glm5-next up-full its FA ubatch-16 and ubatch-64 runs differ in rows 15, 50
# and 54, and those are exactly the rows where fork and upstream differ at ubatch 64), so FA ubatch invariance is
# not required and the FA ubatch-64 fork-vs-upstream comparison is informational.
#
# usage: tests/streamllama/run_acceptance.sh <arch-dir> <fork-build-bin> <upstream-src> <upstream-build-bin> <work-dir>
#   e.g. tests/streamllama/run_acceptance.sh glm5-next build-cpu/bin ../upstream ../build-upstream/bin /tmp/acc
# <arch-dir>/make_tiny.sh <work-dir> <upstream-src> writes the GGUFs and lists them in <work-dir>/models.txt as
# "<file> <compare-with-upstream: 1|0> <must-equal-fork-output-of: file|->".
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ARCH=$1; FBIN=$(cd "$2" && pwd); USRC=$(cd "$3" && pwd); UBIN=$(cd "$4" && pwd); WORK=$5
FORK=$(cd "$HERE/../.." && pwd)
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
c++ -O2 -std=c++17 -I"$FORK/include" -I"$FORK/ggml/include" "$HERE/logits_dump.cpp" -L"$FBIN" -lllama -lggml -lggml-base -Wl,-rpath,"$FBIN" -o "$WORK/dump-fork"
c++ -O2 -std=c++17 -I"$USRC/include" -I"$USRC/ggml/include" "$HERE/logits_dump.cpp" -L"$UBIN" -lllama -lggml -lggml-base -Wl,-rpath,"$UBIN" -o "$WORK/dump-up"
sh "$HERE/$ARCH/make_tiny.sh" "$WORK" "$USRC"
cd "$WORK"
fail=0
while read -r m up same; do
    for fa in 0 1; do
        ./dump-fork "$m" "f-$m-u64-fa$fa.bin" 64 64 8 $fa > "f-$m-u64-fa$fa.log" 2>&1
        ./dump-fork "$m" "f-$m-u16-fa$fa.bin" 64 16 8 $fa > "f-$m-u16-fa$fa.log" 2>&1
        if [ "$fa" = 0 ]; then
            echo "== $m fa=$fa: fork ubatch16 vs fork ubatch64"
            python3 "$HERE/compare_logits.py" "f-$m-u16-fa$fa.bin" "f-$m-u64-fa$fa.bin" || fail=1
        fi
        if [ "$up" = 1 ]; then
            ./dump-up "$m" "u-$m-u64-fa$fa.bin" 64 64 8 $fa > "u-$m-u64-fa$fa.log" 2>&1
            ./dump-up "$m" "u-$m-u16-fa$fa.bin" 64 16 8 $fa > "u-$m-u16-fa$fa.log" 2>&1
            echo "== $m fa=$fa: fork ubatch64 vs upstream ubatch64"
            if [ "$fa" = 0 ]; then
                python3 "$HERE/compare_logits.py" "f-$m-u64-fa$fa.bin" "u-$m-u64-fa$fa.bin" || fail=1
            else
                # informational: see the FA note at the top
                python3 "$HERE/compare_logits.py" "f-$m-u64-fa$fa.bin" "u-$m-u64-fa$fa.bin" || echo "   (informational, FA ubatch64)"
            fi
            echo "== $m fa=$fa: fork ubatch16 vs upstream ubatch16"
            python3 "$HERE/compare_logits.py" "f-$m-u16-fa$fa.bin" "u-$m-u16-fa$fa.bin" || fail=1
            if [ "$fa" = 0 ]; then
                echo "== $m fa=$fa: fork ubatch16 vs upstream ubatch64"
                python3 "$HERE/compare_logits.py" "f-$m-u16-fa$fa.bin" "u-$m-u64-fa$fa.bin" || fail=1
            fi
        fi
        if [ "$same" != "-" ]; then
            echo "== $m fa=$fa: tensor-layout parity on $same (A_log exp may differ by ULPs)"
            python3 "$HERE/compare_logits.py" "f-$m-u64-fa$fa.bin" "f-$same-u64-fa$fa.bin" --tol 1e-4 || fail=1
        fi
    done
done < models.txt
[ $fail = 0 ] && echo "ACCEPTANCE: PASS" || echo "ACCEPTANCE: FAIL"
exit $fail
