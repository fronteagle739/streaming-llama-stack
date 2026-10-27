"""Default-off fork ports; verification is excluded independently of its size."""

from collections import namedtuple
from contextlib import contextmanager
from contextvars import ContextVar
import os

_VERIFY = ContextVar("dsv41_fork_kernel_verify", default=False)


def enabled(kind, n):
    return (n >= 64 and not _VERIFY.get()
            and os.environ.get("DSV41_PREFILL_V41_" + kind, "0") == "1")


@contextmanager
def verification():
    token = _VERIFY.set(True)
    try:
        yield
    finally:
        _VERIFY.reset(token)


# ---------------------------------------------------------------------------
# rb16 attention work decomposition.
#
# One threadgroup holds one query token and COHORT heads, one simdgroup each,
# so the whole cohort shares the rows staged in kv_shared: doubling COHORT
# halves the head-groups and therefore halves the gathered bytes per layer.
# BLOCK is the number of selected IDs staged per pass; kv_shared is
# BLOCK * 128 half4 whatever COHORT is. Both default to the ported geometry,
# so nothing changes unless the variable is set. An unset variable takes the
# default; any set value outside the allowed set is a ValueError, including an
# empty or non-numeric one. There is no silent fallback.
SIMD_WIDTH = 32
MAX_THREADS_PER_THREADGROUP = 1024  # Metal's hard cap; the driver rejects more.
HALF4_BYTES = 8
ATTN_HEAD_DIM = 512
ATTN_COHORT_VAR = "DSV41_V41_ATTN_COHORT"
ATTN_BLOCK_VAR = "DSV41_V41_ATTN_BLOCK"
ATTN_COHORT_VALUES = (4, 8, 16, 32)
ATTN_BLOCK_VALUES = (4, 8, 16)
ATTN_COHORT_DEFAULT = 8
ATTN_BLOCK_DEFAULT = 16

AttnGeometry = namedtuple(
    "AttnGeometry",
    "cohort block threads head_groups grid threadgroup threadgroup_bytes")


def _tuned(name, override, default, allowed):
    """The override, else the environment, else the default; never a fallback."""
    raw = os.environ.get(name) if override is None else override
    if raw is None:
        return default
    value = None
    if isinstance(raw, int) and not isinstance(raw, bool):
        value = raw
    elif isinstance(raw, str):
        try:
            value = int(raw.strip(), 10)
        except ValueError:
            value = None
    if value not in allowed:
        raise ValueError(
            f"{name}={raw!r} is not a valid setting; "
            f"allowed values are {list(allowed)}")
    return value


def attn_cohort(override=None):
    """Heads per threadgroup, one simdgroup each."""
    return _tuned(ATTN_COHORT_VAR, override, ATTN_COHORT_DEFAULT,
                  ATTN_COHORT_VALUES)


def attn_block(override=None):
    """Selected IDs staged per pass; kv_shared is BLOCK * 128 half4."""
    return _tuned(ATTN_BLOCK_VAR, override, ATTN_BLOCK_DEFAULT,
                  ATTN_BLOCK_VALUES)


def attn_threads_per_threadgroup(cohort):
    """32 lanes per head times COHORT heads, refused above Metal's 1024 cap."""
    threads = SIMD_WIDTH * int(cohort)
    if threads > MAX_THREADS_PER_THREADGROUP:
        raise ValueError(
            f"{ATTN_COHORT_VAR}={cohort} needs {threads} threads per "
            f"threadgroup, above the Metal limit of "
            f"{MAX_THREADS_PER_THREADGROUP}")
    return threads


def attn_head_groups(h, cohort):
    """Threadgroups in grid y; a partial cohort still gets a whole group."""
    h, cohort = int(h), int(cohort)
    if h < 1 or cohort < 1:
        raise ValueError(f"heads={h} and cohort={cohort} must both be positive")
    return -(-h // cohort)


def attn_threadgroup_bytes(block):
    """kv_shared, which is BLOCK * 128 half4 and independent of COHORT."""
    return int(block) * 128 * HALF4_BYTES


def attn_gathered_bytes(n, h, k, cohort, head_dim=ATTN_HEAD_DIM,
                        element_bytes=2):
    """Rows pulled per layer: every head-group gathers the same rows again."""
    return (int(n) * attn_head_groups(h, cohort) * int(k)
            * int(head_dim) * int(element_bytes))


def attn_geometry(n, h, b=1, cohort=None, block=None):
    """The whole launch: threads, both template constants and the grid."""
    n, h, b = int(n), int(h), int(b)
    if min(n, h, b) < 1:
        raise ValueError(f"tokens={n}, heads={h}, batch={b} must be positive")
    cohort = attn_cohort(cohort)
    block = attn_block(block)
    threads = attn_threads_per_threadgroup(cohort)
    head_groups = attn_head_groups(h, cohort)
    return AttnGeometry(cohort=cohort, block=block, threads=threads,
                        head_groups=head_groups,
                        grid=(n * threads, head_groups, b),
                        threadgroup=(threads, 1, 1),
                        threadgroup_bytes=attn_threadgroup_bytes(block))
