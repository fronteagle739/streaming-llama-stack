# Stack development plan

The stack provides a configurable local inference launcher and chat endpoint.
It integrates native and streaming backends, optional paired execution,
quality controls, media generation and adapter training/export.

## Validation

Run the CPU suite with `python -B -m pytest -q tests` in an isolated runtime.
Set STUDIO_RUNTIME_ROOT to an existing writable directory and put the checkout,
vendor/studio and vendor/engines on PYTHONPATH. Configure temporary directories
and an explicit host-memory ceiling before running tests or builds.

The CPU suite checks protocol, configuration and resource ownership. Native
help/CLI checks require built binaries and optional training checks require
their documented dependencies. Record unavailable prerequisites explicitly.
GPU, paired execution, long-context, media and throughput acceptance require
separate target-hardware qualification; CPU fixtures do not establish them.

## Remaining engineering work

* Qualify each supported backend and arithmetic profile on its selected device.
* Validate paired execution and state transfer against native output.
* Exercise long prompts, follow-up turns, cancellation and media handoffs.
* Keep generated environment documentation synchronized with source defaults.
* Preserve model identity checks, explicit memory admission and owned cleanup.

Use local configuration for model roots, runtime/cache locations, GPU identity
and memory budgets. Do not commit deployment inventories or private evidence.
