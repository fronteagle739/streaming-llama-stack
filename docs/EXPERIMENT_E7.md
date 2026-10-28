# E7: CPU experts for small row batches

Reject the tested8-thread CPU expert component. GPU compute wins at every
measured row count, and CPU/GPU arithmetic differs. No engine preset changes.
A CPU/GPU hybrid for up to12K rows still needs native routing/integration;
this component does not establish a hybrid speedup or full-model quality.

One actual Q4_K expert from layer0 is read directly from the original GGUF
into19,906,560 bytes of bounded RAM. Gate/up/down matrices use the native
shapes5120x2304/2304x5120, F16 gate/up and intermediate round trips, gate
upper clamp10, up clamp[-10,10],SiLU and route weight1. The shared expert,
six-expert sum, real routed inputs and engine scheduling are outside scope.
The deterministic input is identical in both backends. The sole component
variable is8-thread GGML CPU versus CUDA; weights and inputs are resident
before timing. Two warmups and three repeats, median compute plus sync:

| Rows | CPU ms | GPU ms | CPU/GPU | Relative L2 difference | Max absolute difference |
|---|---:|---:|---:|---:|---:|
| 1 | 0.0980 | 0.0867 | 1.13 | 0.01680 | 0.002060 |
| 32 | 2.5233 | 0.1248 | 20.22 | 0.02717 | 0.004656 |
| 128 | 10.0516 | 0.1726 | 58.24 | 0.02610 | 0.004656 |
| 256 | 20.1525 | 0.2061 | 97.78 | 0.02627 | 0.004659 |

Outputs are finite, but they are not exact. Quantized GEMM/reduction paths
can differ; no recall/KL acceptance is inferred from this local norm test.
The component excludes streaming, weight uploads and transfers. It therefore
cannot predict end-to-end TTFT, CPU/GPU overlap, or the cost of transferring
routed activations. No TTFT or prompt-throughput claim is made.

Source, extent manifest, binary pin and guarded receipt are in
`examples/perfexp/e7/`. Build the driver with the existing GGML CPU/CUDA
libraries and CUDA13.2. Run under the shared exclusive lease and OwnedProcess:
3GiB host cap,0.5GiB physical floor,512MiB desktop reserve, paging stops.
Peak RSS438,042,624 bytes, minimum available RAM14,400,839,680 bytes,
confirmed graceful cleanup, healthy Studio and Remote Link afterwards.
No weights were copied to disk or changed. Bounded paging observed0;
Windows kernel ETW attribution was unavailable.
