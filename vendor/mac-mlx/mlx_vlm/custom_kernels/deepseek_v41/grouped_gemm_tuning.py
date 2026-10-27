"""Benchmark-only configurations and launch accounting; no MLX dependency."""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class GemmConfig:
    tiles: tuple = (32, 64, 32)
    vector_decode: bool = False
    native_loads: bool = False
    contiguous: bool = False
    padding: int = 8
    unroll: int = 1

    def validate(self, itemsize):
        bm, bn, bk = self.tiles
        if bm not in (16, 32, 64, 128) or bn not in (32, 64, 128) or bk not in (32, 64):
            raise ValueError("supported sweep: BM16/32/64/128, BN32/64/128, BK32/64")
        if self.padding not in (0, 8, 16) or self.unroll not in (1, 2, 4):
            raise ValueError("padding must be 0/8/16 and unroll must be 1/2/4")
        if (bm + bn) * (bk + self.padding) * itemsize > 32768:
            raise ValueError("configuration exceeds the M3 32 KiB threadgroup memory limit")


LEGACY = GemmConfig()
FAST = GemmConfig(vector_decode=True, native_loads=True, contiguous=True, unroll=4)
CONFIGS = {
    "legacy": LEGACY,
    "vector": replace(LEGACY, vector_decode=True),
    "native": replace(LEGACY, native_loads=True),
    "fast": FAST,
    "bk64": replace(FAST, tiles=(32, 64, 64)),
    "bn128": replace(FAST, tiles=(32, 128, 32)),
    "unroll2": replace(FAST, unroll=2),
    "pad0": replace(FAST, padding=0),
    "pad16": replace(FAST, padding=16),
    "rows64": replace(FAST, tiles=(64, 64, 32)),
    "rows128": replace(FAST, tiles=(128, 64, 32)),
}


def legacy_bank_model(ld, itemsize):
    """Hypothetical 32 x 4-byte banks, distinct words per bank per scalar load.

    This is an address audit, NOT a claim about Apple's bank implementation or
    how simdgroup_load lowers. Same-word lane requests count once (broadcast).
    """
    result = {}
    for operand in ("x", "w"):
        worst = 0
        for j in (0, 1):
            banks = [set() for _ in range(32)]
            for lane in range(32):
                fm = ((lane // 4) & 4) + ((lane // 2) % 4)
                fn = ((lane // 4) & 2) * 2 + (lane % 2) * 2
                index = fm * ld + fn + j if operand == "x" else (fn + j) * ld + fm
                word = index * itemsize // 4
                banks[word % 32].add(word)
            worst = max(worst, max(map(len, banks)))
        result[operand + "_max_distinct_words_per_bank"] = worst
    return dict(assumption="hypothetical 32 banks x 4 bytes; original scalar lane loads only", **result)


def launch_info(config, counts, n, k, itemsize):
    """Source-level resources, not a compiler occupancy/residency estimate."""
    config.validate(itemsize)
    bm, bn, bk = config.tiles
    rows = sum(int(c) for c in counts)
    tasks = sum((int(c) + bm - 1) // bm for c in counts)
    capacity = rows // bm + len(counts)
    bands = (n + bn - 1) // bn
    return dict(
        tiles=config.tiles, padding_elements=config.padding,
        threads_per_threadgroup=128, simdgroups_per_threadgroup=4,
        threadgroup_memory_bytes=(bm + bn) * (bk + config.padding) * itemsize,
        legacy_load_bank_address_model=legacy_bank_model(bk + config.padding, itemsize),
        registers_per_lane="unavailable: MLX runtime compiler exposes no register count",
        accumulator_fp32_values_per_lane=2 * (bm // 16) * (bn // 16),
        worklist_length=tasks, worklist_capacity=capacity, column_bands=bands,
        threadgroups_launched=capacity * bands, useful_threadgroups=tasks * bands,
        sentinel_threadgroups=(capacity - tasks) * bands,
        k_steps=(k + bk - 1) // bk, barriers_per_k_step=2,
        useful_flops=2 * rows * n * k,
        tiled_flops=2 * tasks * bm * bands * bn * ((k + bk - 1) // bk) * bk,
    )
