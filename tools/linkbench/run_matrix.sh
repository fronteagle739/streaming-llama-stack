#!/usr/bin/env bash
# Runs the linkbench matrix from the RTX against servers on the Mac.
# Usage: run_matrix.sh LINKBENCH_EXE MAC_SSH MAC_IP OUT [ITERS] [GAP_US]
# Mac side must already run: linkbench server tcp 192.168.3.1:47801, udp :47802,
# tcp 127.0.0.1:47811, and
# relay_probe.py lan :47803 -> 47811, relay_probe.py loopback -> 47811 (port in relay_lo.log).
# SECTIONS=relay,loopback limits the run (default: all).
set -u
B=$1; MAC=$2; IP=$3; OUT=$4; IT=${5:-10000}; GAP=${6:-200}
C="--iters $IT --warmup 500 --gap-us $GAP"
T="--nodelay --quickack --spin --prio --cpu 6"
run() { echo "# $(date +%T) $*" >> "$OUT"; "$B" client "$@" $C >> "$OUT" 2>&1; }
SECTIONS=${SECTIONS:-main,relay,loopback}
has() { case ",$SECTIONS," in *",$1,"*) return 0;; esac; return 1; }
for dir in fwd rev; do
  R=""; [ $dir = rev ] && R="--reverse"
  if has main; then
  run --proto tcp --host $IP --port 47801 $R --label "tcp-plain-$dir"
  run --proto tcp --host $IP --port 47801 $R --nodelay --label "tcp-nodelay-$dir"
  run --proto tcp --host $IP --port 47801 $R --nodelay --quickack --label "tcp-nodelay-qa-$dir"
  run --proto tcp --host $IP --port 47801 $R $T --label "tcp-tuned-$dir"
  run --proto tcp --host $IP --port 47801 $R $T --bufs 4194304 --label "tcp-tuned-4mbuf-$dir"
  run --proto udp --host $IP --port 47802 $R --label "udp-block-$dir"
  run --proto udp --host $IP --port 47802 $R --spin --prio --cpu 6 --label "udp-spin-$dir"
  run --proto udp --host $IP --port 47802 $R --spin --prio --cpu 6 --chunk 8972 --bufs 4194304 --label "udp-spin-ipfrag-$dir"
  fi
  if has relay; then
  run --proto tcp --host $IP --port 47803 $R --nodelay --label "relay-nodelay-$dir"
  run --proto tcp --host $IP --port 47803 $R $T --label "relay-tuned-$dir"
  fi
done
if has loopback; then
LO=$(ssh $MAC 'grep -o "RELAY_PORT=[0-9]*" linkbench-scratch/relay_lo.log | cut -d= -f2')
for f in "--nodelay" "--nodelay --spin --prio"; do
  tag=$(echo "$f" | tr -d ' -')
  echo "# $(date +%T) mac loopback $f" >> "$OUT"
  ssh $MAC "cd linkbench-scratch && ./linkbench client --proto tcp --host 127.0.0.1 --port 47811 $f $C --label mac-loopback-$tag && ./linkbench client --proto tcp --host 127.0.0.1 --port $LO $f $C --label mac-loopback-relay-$tag" >> "$OUT" 2>&1
done
fi
echo "# $(date +%T) DONE" >> "$OUT"
