# Environment behaviour samples

Run on2026-10-22 on Windows x64. Stack parent:8d6dd1bb3b3e361402e9a558a37c3d0879697932 plus the evolving working changes published with this example. Runtime identities and publication source hashes are recorded in SOURCE_IDENTITY.json. These samples use CPU only.

The reused image runtime's real memory watchdog received floor5 and stayed alive (0.578s,27,881,472B peak RSS). It received floor64 and exited137 with MEM_GUARD_TRIP (0.344s,25,669,632B). The outer guardian enforced1GiB RSS and4GiB available RAM in both cases. This tests the engine's memory guard, not image generation.

The actual native executable received bias0 and reached a deliberate missing-model error. Bias0.35 instead triggered its CUDA-only routing refusal on CPU. The launcher flag also removed a conflicting configured engine argument. These are startup behaviour checks, not inference or speed measurements.

Edit the configuration path, using a1GiB host budget and4GiB available floor. Set job_commit_gib to4 for the native CPU control. Commands actually used, with the local paths replaced:

```text
python -B tests/prove_image_environment.py --config <edit-this-config.json> --expect admitted --receipt <edit-this-admitted.json> --mem-guard-min-free-gb 5
python -B tests/prove_image_environment.py --config <edit-this-config.json> --expect refused --receipt <edit-this-refused.json> --mem-guard-min-free-gb 64
python -B tests/prove_native_environment.py --config <edit-this-config.json> --expect model-open --receipt <edit-this-model-open.json> --ds4-cache-affinity-bias 0
python -B tests/prove_native_environment.py --config <edit-this-config.json> --expect cuda-required --receipt <edit-this-cuda-required.json> --ds4-cache-affinity-bias 0.35
```

The JSON receipts preserve each child's output. GPU prefill and Mac-variable behaviour samples remain separate gates.
