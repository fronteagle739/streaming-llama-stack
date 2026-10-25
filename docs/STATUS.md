# Stack status

Front Eagle maintains a portable launcher, local chat router, native inference
integration, optional paired receiver, media tools and adapter workflows.

CPU fixtures exercise configuration, protocol handling, memory admission,
ownership and cleanup. They do not qualify model output, GPU performance,
paired inference, long-context operation or media generation on a new system.
Build and qualify each selected backend on its target hardware before use.

Model weights, credentials, runtime paths and device selection are supplied by
local configuration. Generated runtime files and private receipts stay outside
version control. See README.md and USER_GUIDE.md for commands and settings.

## CPU validation, 2026-10-25

The original release baseline had 414 passing tests, 3 failures and 16 skips.
After portable-default changes, the suite has 418 passing tests, 2 failures and
16 skips, with 101 passing subtests. Three added CPU tests cover configurable
storage policy and fail-closed GPU identity/capability checks. Regenerating the
environment inventory resolves its existing baseline failure.

The remaining baseline failures are the unavailable native CLI test executable
and the existing quality-tool schema validator error. The suite is not green.
No new failure was introduced. The release scanner and generated-guide check
pass. This cleanup did not run engines or qualify the CUDA source changes on
hardware; native build and target-device qualification remain required.
