# Backend installation

The stack retains the custom decoder sources and installs external dependencies
at pinned revisions. Build inputs, environments, compiler outputs and receipts
live under `runtime_root`; model weights and host settings are separate.
Installation does not require Studio or a Studio virtual environment.

On Apple Silicon, install Python 3.12 and the Xcode command-line tools, including
the Metal compiler. Use the stack's Python interpreter for these commands:

```sh
python3.12 scripts/bootstrap.py setup-backend --backend mlx-receiver
python3.12 scripts/bootstrap.py setup-backend --backend llama-decode
python3.12 scripts/bootstrap.py setup-backend --backend llama-decode-upstream
python3.12 scripts/bootstrap.py setup-backend --backend ds4-peer
python3.12 scripts/bootstrap.py setup-backend --backend mference
python3.12 scripts/bootstrap.py setup-backend --backend mlx-serve
python3.12 scripts/bootstrap.py setup-backend --backend omlx
```

Pass a private config using `--config` before the command. `--backend all` installs
all registered Mac backends. Keep at least 16 GiB free on the runtime volume;
large compiler caches and multiple backends may require more. Compiler jobs are
limited to one or two and run under the stack's host memory guardian.

| Backend | Installation |
| --- | --- |
| `mlx-receiver`, `mlx-lm`, `mlx-vlm`, `mtp-shim` | Retained custom MLX-VLM, MLX 0.32.2 and GLM native kernels in `decode-venv`. |
| `llama-decode`, `llama-server` | Retained streaming llama source, Metal and Accelerate. |
| `llama-decode-upstream` | Pinned upstream source plus the retained KV/token receiver overlay. WIR is unavailable in this overlay. |
| `ds4` | DSV4.1 native state receiver. Existing `mac-setup` also performs its model and wired memory checks. |
| `ds4-peer` | Earlier DS4P protocol, built from the retained Windows fork's portable Metal sources. This is a distinct protocol from the DSV4.1 state broker. |
| `mference` | Server-only Swift package with pinned package dependencies; Swift 6.1 or newer. |
| `mlx-serve` | Pinned server, DS4, OpenCode, MLX, MLX-C, llama, Jinja and webp inputs, private dependency prefixes and Zig 0.17.0. Requires macOS/Metal 26.2 or newer. |
| `omlx` | Separate `omlx-venv`, MLX 0.32.4, nanobind 3.1.0, pinned model libraries and compiled native kernels. It does not replace the custom MLX environment. |

An OpenAI HTTP backend performs its own prefill. Select a KV/token receiver or
the matching DS4 protocol for RTX prefill and Mac decode.

For an offline Mac, fetch source archives on a connected machine:

```sh
python3.12 scripts/bootstrap.py setup-backend --backend all --fetch-only --cache-dir /path/to/source-cache
```

Transfer that cache, including archive `.json` receipts, and a wheelhouse containing
the packages in the chosen `locks/mac-*.txt` profile. Include CMake 4.4.4,
setuptools 84.2.1, wheel 0.48.0 and, for mlx-serve, ziglang 0.17.0. Then install:

```sh
python3.12 scripts/bootstrap.py setup-backend --backend mlx-serve --offline \
  --cache-dir /path/to/source-cache --wheelhouse /path/to/wheels
```

Downloads are verified before extraction. A divergent cache or build snapshot
is preserved and reported instead of overwritten. Backend receipts record source
and binary hashes. `--check` refuses changed or unverified artifacts.

`--build-only` compiles without model inference or GPU smoke checks and leaves
verification pending. Run setup without that option before normal launch.
Installation and CLI checks establish that an engine is installed; they do not
qualify a model, cross-machine state handoff, output quality or performance.

On Windows, `install --all` and `build.ps1` prepare the retained RTX engines and
WebUI. Install Visual Studio C++ Build Tools, the CUDA toolkit, Ninja and Node.js
first. The SDK installer fetches the pinned DirectStorage redistributable rather
than relying on files in Studio. Choose a CUDA architecture matching the GPU.

See [the backend audit](BACKEND_SOURCE_AUDIT.md) for qualification status and
existing measurements, and [paired examples](../examples/dual-machine/README.md)
for model paths, interface settings and launch order.
