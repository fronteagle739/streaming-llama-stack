# Single-engine RAM experts: Round 2

Continuous victim admission caused the Round 1 slowdown. At Max quality, the original 3.49 GiB cache spent 135.021 ms/token admitting about 95 experts, including 3.935 ms in explicit device synchronization. Uploading RAM hits took another 13.985 ms/token. The whole cache-off native step took 136.823 ms. At Max speed, admissions cost 16.652 ms, uploads 10.049 ms and residency scans 4.493 ms/token. These diagnostic intervals contained 155 warm steps; their binary hashes were not saved before supersession.

The fix selects a bounded hot set after 32 observation passes and fills each selected payload once. A direct key index replaces linear RAM scans. Payloads remain locked and uploads use the existing pinned staging buffers. CPU mode borrows RAM payloads while the existing ds4 row pool and llama Q4_K CPU dot kernels overlap the GPU experts. The final row is combined before advancing the layer. See [design and controls](RAMTIER.md).

The cache stays off by default. No tested tier provides a consistent gain across the dial. Mix uses immediate measured CPU/upload cost; it chose CPU for essentially all warm RAM hits and does not account for future GPU promotion. A positive observation in one Max quality run does not establish a general speedup.

## Full WebUI measurements

Original Q4_K/Q8_0 model, 49,553-token recall and reused-prefix follow-up, then 1,000 tokens at each dial endpoint. Each warm interval contains 899 evaluated steps after the first 100. Historical baseline: 6.8118 / 18.1550 tok/s. All accepted Round 2 points use identical binaries and a fixed 640-slot VRAM L2. Admission ceilings vary with available desktop RAM; exact ceilings and memory observations are in the [machine-readable results](../examples/ramtier/round2-summary.json). The last off control matches the middle/empty tier ceilings. The owner correction sets all three large tests explicitly to 10 GiB (10240 MiB); complete experts round the actual allocation down slightly. Auto-sizing remains uncapped. Earlier auto trials are superseded; pre-initialization CPU/mix results failed correctness and are excluded. Every final point uses the cooperative guardian. Earlier controls have separate provenance.

| Mode | RAM GiB | Quality tok/s | Speed tok/s | Quality L1 / L2 / RAM / SSD % | Speed L1 / L2 / RAM / SSD % |
|---|---:|---:|---:|---|---|
| Off | 0.00 | 6.86965 | 18.25601 | 35.85 / 24.01 / 0.00 / 40.14 | 54.39 / 42.74 / 0.00 / 2.86 |
| Empty tier | 3.49 | 6.74119 | 18.07504 | 35.85 / 24.01 / 0.00 / 40.14 | 54.39 / 42.74 / 0.00 / 2.86 |
| Upload | 3.49 | 6.78797 | 17.34247 | 35.85 / 24.01 / 1.36 / 38.79 | 53.59 / 42.77 / 0.64 / 3.01 |
| CPU | 3.49 | 6.81298 | 14.57991 | 33.96 / 24.31 / 3.00 / 38.74 | 47.52 / 40.75 / 9.42 / 2.31 |
| Mix | 3.49 | 6.84279 | 15.80330 | 35.08 / 24.21 / 2.91 / 37.80 | 47.56 / 40.17 / 9.72 / 2.55 |
| Upload 10 GiB setting | 9.99 | 6.49141 | 15.22702 | 35.85 / 24.01 / 3.55 / 36.59 | 52.92 / 41.52 / 2.29 / 3.27 |
| CPU 10 GiB setting | 9.99 | 6.92617 | 13.21849 | 32.73 / 23.66 / 7.36 / 36.26 | 41.99 / 32.12 / 23.89 / 2.01 |
| Mix 10 GiB setting | 9.99 | 6.94433 | 12.08272 | 35.16 / 23.57 / 6.85 / 34.42 | 43.86 / 30.89 / 23.39 / 1.86 |

RAM percentages combine uploads and CPU service; the JSON separates them. Upload retains exact Max quality output. CPU kernel rounding can change a sampled token and hence later inputs; the CPU/mix free-running rows are not comparisons of identical routed work. The separate teacher-forced check below controls every input. These are bounded single-run observations.

## Per-token timing

| Mode | Dial | Native ms | DirectStorage ms | Admission ms | Upload ms | Residency ms | CPU service ms | CPU wait ms | CPU transfer + merge ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Off | Quality | 144.890 | 100.907 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Off | Speed | 54.022 | 9.140 | 0.000 | 0.000 | 0.037 | 0.000 | 0.000 | 0.000 |
| Empty tier | Quality | 147.666 | 103.059 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Empty tier | Speed | 54.561 | 9.363 | 0.000 | 0.000 | 0.091 | 0.000 | 0.000 | 0.000 |
| Upload | Quality | 146.642 | 97.742 | 0.010 | 4.860 | 0.000 | 0.000 | 0.000 | 0.000 |
| Upload | Speed | 56.901 | 9.650 | 0.000 | 2.275 | 0.098 | 0.000 | 0.000 | 0.000 |
| CPU | Quality | 146.102 | 99.862 | 0.009 | 0.000 | 0.020 | 3.668 | 0.167 | 1.172 |
| CPU | Speed | 67.799 | 11.056 | 0.000 | 0.000 | 0.142 | 11.025 | 6.192 | 3.095 |
| Mix | Quality | 145.452 | 99.098 | 0.020 | 0.000 | 0.019 | 3.550 | 0.161 | 1.171 |
| Mix | Speed | 62.517 | 8.678 | 0.000 | 0.000 | 0.126 | 10.443 | 5.610 | 2.798 |
| Upload 10 GiB setting | Quality | 153.367 | 96.537 | 0.023 | 12.654 | 0.000 | 0.000 | 0.000 | 0.000 |
| Upload 10 GiB setting | Speed | 64.900 | 11.406 | 0.002 | 8.299 | 0.103 | 0.000 | 0.000 | 0.000 |
| CPU 10 GiB setting | Quality | 143.694 | 95.640 | 0.069 | 0.000 | 0.045 | 8.718 | 0.503 | 2.687 |
| CPU 10 GiB setting | Speed | 74.874 | 7.796 | 0.000 | 0.000 | 0.174 | 25.264 | 15.498 | 6.235 |
| Mix 10 GiB setting | Quality | 143.248 | 92.386 | 0.080 | 0.000 | 0.047 | 10.413 | 0.794 | 2.741 |
| Mix 10 GiB setting | Speed | 81.887 | 7.186 | 0.000 | 0.000 | 0.227 | 29.792 | 19.798 | 6.725 |

Admission includes its synchronization time. CPU service overlaps the GPU/DirectStorage work, so the columns are not additive. CPU wait is only the exposed wait after scheduling the GPU experts. The JSON also reports hook, DirectStorage submit/fence/verification/CUDA time and per-tier counters.

## Controlled correctness and zero-hit overhead

CPU and mix remain experimental. After CPU initialization was fixed, a 16-row same-activation hybrid/all-GPU check reached maximum relative RMS 2.24668e-7. At 3.5 GiB, the full sequence check still differs: CPU mean/max KL 0.00437576/0.09530794, top-1 255/256; mix mean/max KL 0.00623824/0.15609729, top-1 253/256. Both fail the agent screening criterion of max KL below 1e-5 and top-1 at least 99%; the owner did not prescribe those numeric thresholds. Small arithmetic differences may be amplified by quantization and MoE routing, but this is not proof that every sequence difference is harmless rounding. An all-GPU control requesting the H16 reduction was bit-exact; actual dispatch was not instrumented, so it did not establish a tolerance envelope. These modes are not qualified as bit-exact Max quality and are not defaults.

The first full-vector check exposed a separate CPU integration bug: the GPU-focused producer never initialized llama CPU FP16 conversion tables, so its SIMD Q4_K dots returned zero. CPU max KL was 1.46791 with 91.40625% top-1 agreement; mix max KL 1.58619 with 91.796875%. All of those CPU/mix timings were discarded. The fix calls ggml_cpu_init before registration and requires a known-answer dot result. An opt-in LLAMADECODE_HOST_CPU_VERIFY compares hybrid rows with all-GPU rows on identical activations; final pointwise error is recorded in the JSON. Explicit stream fences order CPU activation reads and scratch reuse. All final WebUI and forced runs use the corrected build with diagnostic extra work disabled.

The existing native decode_forced path appends 256 identical tokens at bias 0, writes every full-vocabulary logit vector, then repeats a 256-token speed interval. Both CPU/mix tier sizes are checked against the same off reference. All input identities are checked before comparing vectors; all forced timings below exclude the first 100 steps.

```json
{
  "forced_3_5_gib": {
    "zero": {
      "quality": {
        "vectors": 256,
        "mean_kl": 0.0,
        "max_kl": 0.0,
        "top1_agreement": 1.0,
        "max_abs_logit": 0.0,
        "reference_warm_tok_s": 6.984501971416236,
        "candidate_warm_tok_s": 7.20286035529303,
        "raw_slowdown_percent": -3.0315509826083553
      },
      "speed": {
        "vectors": 256,
        "mean_kl": 0.0,
        "max_kl": 0.0,
        "top1_agreement": 1.0,
        "max_abs_logit": 0.0,
        "reference_warm_tok_s": 17.97410937389347,
        "candidate_warm_tok_s": 18.93947725087745,
        "raw_slowdown_percent": -5.097119969027942
      }
    },
    "cpu": {
      "quality": {
        "vectors": 256,
        "mean_kl": 0.004375764523058049,
        "max_kl": 0.09530793951513122,
        "top1_agreement": 0.99609375,
        "max_abs_logit": 2.527545928955078,
        "reference_warm_tok_s": 6.984501971416236,
        "candidate_warm_tok_s": 7.247435364075009,
        "raw_slowdown_percent": -3.627950846752126,
        "strict_sequence_screen_pass": false
      },
      "speed": {
        "vectors": 256,
        "mean_kl": 0.0028343125875376633,
        "max_kl": 0.6539617298403493,
        "top1_agreement": 0.99609375,
        "max_abs_logit": 8.37492847442627,
        "reference_warm_tok_s": 17.97410937389347,
        "candidate_warm_tok_s": 13.992980362720802,
        "raw_slowdown_percent": 28.450901151687003
      }
    },
    "mix": {
      "quality": {
        "vectors": 256,
        "mean_kl": 0.006238236484258442,
        "max_kl": 0.15609729269497769,
        "top1_agreement": 0.98828125,
        "max_abs_logit": 5.607177257537842,
        "reference_warm_tok_s": 6.984501971416236,
        "candidate_warm_tok_s": 7.337099133387767,
        "raw_slowdown_percent": -4.805675316107195,
        "strict_sequence_screen_pass": false
      },
      "speed": {
        "vectors": 256,
        "mean_kl": 0.0023747201076236312,
        "max_kl": 0.4966067297224692,
        "top1_agreement": 1.0,
        "max_abs_logit": 8.603780269622803,
        "reference_warm_tok_s": 17.97410937389347,
        "candidate_warm_tok_s": 15.896434340552268,
        "raw_slowdown_percent": 13.070069606999791
      }
    }
  },
  "forced_10_gib": {
    "cpu": {
      "quality": {
        "vectors": 256,
        "mean_kl": 0.00567974435816054,
        "max_kl": 0.15880987403365882,
        "top1_agreement": 0.984375,
        "max_abs_logit": 6.724740982055664,
        "reference_warm_tok_s": 6.984501971416236,
        "candidate_warm_tok_s": 7.337066416947639,
        "raw_slowdown_percent": -4.8052508386325465,
        "strict_sequence_screen_pass": false
      },
      "speed": {
        "vectors": 256,
        "mean_kl": 0.004618830190442622,
        "max_kl": 1.1340402958212428,
        "top1_agreement": 0.99609375,
        "max_abs_logit": 8.782349586486816,
        "reference_warm_tok_s": 17.97410937389347,
        "candidate_warm_tok_s": 12.999895027944408,
        "raw_slowdown_percent": 38.26349624559702
      }
    },
    "mix": {
      "quality": {
        "vectors": 256,
        "mean_kl": 0.006611997366672491,
        "max_kl": 0.217349771697234,
        "top1_agreement": 0.98828125,
        "max_abs_logit": 8.065932273864746,
        "reference_warm_tok_s": 6.984501971416236,
        "candidate_warm_tok_s": 7.458409598378405,
        "raw_slowdown_percent": -6.354003768648009,
        "strict_sequence_screen_pass": false
      },
      "speed": {
        "vectors": 256,
        "mean_kl": 0.00787268871890187,
        "max_kl": 1.9350583960453063,
        "top1_agreement": 1.0,
        "max_abs_logit": 9.373231649398804,
        "reference_warm_tok_s": 17.97410937389347,
        "candidate_warm_tok_s": 12.888530681974649,
        "raw_slowdown_percent": 39.45817267620191
      }
    }
  },
  "zero_hit_webui": {
    "max-quality": {
      "raw_slowdown_percent": 1.9056701516237728,
      "non_dstorage_extra_ms": 0.6245561735261305,
      "non_dstorage_overhead_percent": 0.43105485374078345
    },
    "max-speed": {
      "raw_slowdown_percent": 1.0012294478499273,
      "non_dstorage_extra_ms": 0.31606896551723906,
      "non_dstorage_overhead_percent": 0.5850797759216854
    }
  }
}
```

The final 1,000-token WebUI empty tier was 1.90567% slower at Max quality and 1.00123% slower at Max speed. Its added non-DirectStorage native time was 0.43105%/0.58508%. Thus the long raw WebUI pair does not independently meet the 1% target. The controlled native pair below does; these observations do not establish a universal 1% latency guarantee.

The final controlled empty-tier pair meets the <=1% raw slowdown criterion at both endpoints, with bit-for-bit identical full-vocabulary logits. Its negative apparent overhead is run-to-run variation, not a claimed speedup. The full WebUI empty-tier pair and the controlled native pair are both reported; differences in DirectStorage wait must not be hidden as cache bookkeeping. The early off control measured 7.15949 / 18.54121 tok/s; against it, the first empty-tier raw rates were 2.6% / 1.2% slower, with most extra time in DirectStorage.

Upload and empty-tier recall/follow-up/quality logits were bit-for-bit equal to off in the sampled positions recorded in JSON. The full forced CPU check evaluates the kernel arithmetic on identical prefixes instead of treating different sampled wording as a correctness verdict. The GPU byte oracle checks exact host/device bytes, mixed extents, LRU replacement, index invalidation and resize/clear; its receipt is in the JSON.

## Memory, lifecycle and exclusions

Minimum available RAM across accepted WebUI points: 3.47 GiB. Runtime floor remains 512 MiB and desktop reserve 512 MiB, with paging stops intact. Every accepted native cleanup is graceful, has exit 0, no owned process remaining and zero native budget breaches. All 10 GiB setting points pass the larger append with tier release/recommit/regrowth. Global pagefile growth is not process attribution or a zero-paging claim.

Some starts were refused before model launch as desktop headroom changed; admissions also yielded to foreign GPU/deployment holds. An expired inherited lease reference was refused before the forced oracle and removed before obtaining a fresh lease. An earlier maximum CPU attempt completed its decode replies but reached the early aggregate RSS margin before append completion. It cancelled cooperatively with available RAM above 2.8 GiB and native/hard budgets intact; it lacks the normal graceful-completion certificate and is excluded. The retry reserves more controller headroom inside physical admission. A second maximum CPU retry was force-stopped by the pre-existing paging guardian at 270,200,832 bytes global pagefile growth despite 3.16 GiB available RAM. This violated the no-force-kill rule. The early task watcher saw 194,703,360 bytes against a different psutil baseline and did not protect it. The trial is excluded. Before further GPU work, single-engine guardians were changed to cooperative cancellation and local shutdown, kill-on-close was removed for those trees, stalled retirement retains ownership, and the early watcher was aligned to EnumPageFilesW with a running minimum. Paging limits remain intact. Dead cooperative guardians are never automatically treated as evidence of GPU retirement; their leases remain held for inspection. The focused CPU/lifecycle suite passed 89 tests, including synthetic paging cancellation, stalled ownership and dead-guardian recovery. An actual GPU cancellation oracle exited 130 at the abort boundary with all guardian forced_kill flags false and cleanup confirmed; its old native shutdown-path label mentions terminate because the producer had already exited. The Round 1 force-stop incident also remains disclosed in the original report.

Measurements preceded the final source commit, so launcher provenance retains the branch predecessor. Binary hashes and source-file hashes bind the actual tested build; the final delivery report supplies the pushed branch SHA. No model/store edits or replacement weights were used.
