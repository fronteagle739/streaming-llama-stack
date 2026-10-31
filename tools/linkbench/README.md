# linkbench

Serial request/response round-trip benchmark for the RTX <-> Mac Studio link
(TCP plain/tuned, UDP with its own seq/ack, both directions, and the Mac
native relay hop). Usage is in the header of `linkbench.c`.

    # RTX
    powershell -File tools/linkbench/build.ps1 L:\scratch\linkbench\bin\linkbench.exe
    # Mac
    cc -O2 -o linkbench tools/linkbench/linkbench.c
    ./linkbench server --proto tcp --bind 192.168.3.1 --port 47801 &
    ./linkbench server --proto udp --bind 192.168.3.1 --port 47802 &
    ./linkbench server --proto tcp --bind 127.0.0.1  --port 47811 &
    python3 relay_probe.py lan 192.168.3.1 47803 47811 &   # copy slstack/native_relay.py next to it
    python3 relay_probe.py loopback 47811 &
    # RTX: full matrix, then a table
    bash tools/linkbench/run_matrix.sh linkbench.exe edstudio@192.168.3.1 192.168.3.1 out.txt 10000 200
    python tools/linkbench/summarize.py out.txt

Only the Mac listens, so the RTX never opens an inbound firewall port;
`--reverse` makes the Mac the timing side. Results from 2026-10-31 are in
`results/20261031/`.
