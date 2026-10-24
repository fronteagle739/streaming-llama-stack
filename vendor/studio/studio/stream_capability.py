# Derived from Studio backend/studio/stream_capability.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1; copied unchanged below.
"""Single source of truth for how generation architectures handle the GUI weight-streaming toggle.

``GenerationParams.weight_stream`` and the 4-way ``load_mode`` now distinguish real streaming from
bounded-memory fallbacks. CPU-offload is the worst backup path: useful only for diagnostics while native
streamers are being built, and never allowed to count as "streaming implemented" in audits.

Capabilities:

  - ``native``: true per-block/per-layer weight streaming, such as DirectStorage NVMe->GPU, Wan lazy
    high/low denoisers, the LTX streamed loop, the Cosmos DS engine, or the Z-Image block streamer.
    The target is minimum RAM/VRAM with full throughput, especially for batched generations that
    amortize stream setup and reuse layer fetches across the batch.
  - ``direct-resident``: direct D-drive checkpoint load to GPU, no CPU-offload/staged fallback.
    This is not a block streamer, but it is the high-performance path for small checkpoints that fit
    the RTX card and should not be treated as migration debt.
  - ``offload``: compatibility fallback only, using diffusers CPU-offload/staged residency. This is
    bounded VRAM, but host-staged and still migration debt.
  - ``declared-unsupported``: no streaming path in this build. The GUI/model note must say so.

The classifier mirrors ``adapter.RealCosmosBackend.generate`` dispatch order. Keep the matrix test in
``tests/test_stream_capability.py`` aligned whenever a new arch lands.
"""
from __future__ import annotations

NATIVE = "native"
DIRECT = "direct-resident"
OFFLOAD = "offload"
UNSUPPORTED = "declared-unsupported"

VALID_CAPABILITIES = (NATIVE, DIRECT, OFFLOAD, UNSUPPORTED)

# Ordered (prefix, capability) rules. The first matching prefix wins.
_PREFIX_RULES: tuple[tuple[str, str], ...] = (
    ("moss-image", NATIVE),       # shared DirectStorage block binder, native MOSS layer names
    ("minimax-h3", NATIVE),       # dense-DiT DirectStorage block streamer + separate VAE/mux stages
    ("magi-2", NATIVE),           # single-GPU stream-sdpa: whole-model layer-ring NVMe stream (preview MoE) + streamed Qwen3.5 text-encode + SDPA attn (magi2_backend); validated 272p t2v/i2v mp4 on one 5090
    ("magi", OFFLOAD),            # other magi variants: legacy torchrun/offload until a native streamer lands
    ("bernini", NATIVE),          # Wan-derived streamed backend
    ("scail", NATIVE),            # SCAIL native block streamer; clean SDPA-fallback attention on SM120 like Wan/Bernini
    ("domainshuttle", NATIVE),    # dual-expert DirectStorage block streamer, one expert active at a time
    ("domain-shuttle", NATIVE),
    ("wan", NATIVE),              # lazy high/low denoisers, one expert active at a time
    ("hunyuan", NATIVE),          # separate streamed subprocess
    ("ideogram", DIRECT),         # separate-venv direct resident/disposable route with lazy VAE; no CPU-offload
    ("joyai", NATIVE),            # routes to the streamed LTX loop
    ("ltx", NATIVE),              # streamed 22B loop
    ("z-image", NATIVE),          # per-block streamer
    ("zimage", NATIVE),
    ("flux2", DIRECT),            # direct resident/disposable on the 5090; no CPU-offload fallback
    ("flux-2", DIRECT),
    ("klein", DIRECT),
    ("ultratex", NATIVE),         # Native FLUX.2 block ring; direct residency also supported.
    ("qwen-image-2.1", NATIVE),   # DirectStorage block binding, optional INT8 active-block dequantization
    ("anima", NATIVE),            # CosmosTransformer3D block streamer
    ("sdxl", DIRECT),             # direct D-drive -> VRAM resident load; no CPU-offload fallback
    ("sd-xl", DIRECT),
    ("sd_xl", DIRECT),
    ("glm", NATIVE),              # DirectStorage NVMe->GPU stream
    ("krea2-native", NATIVE),     # meta-init + per-block safetensors bind/evict streamer
    ("krea2-raw", NATIVE),        # original HF layout: validate source, bind/evict one meta block
    ("krea2-turbo", NATIVE),
    ("krea2", NATIVE),
    ("krea", NATIVE),
    ("boogu", NATIVE),            # base safetensors DiT: meta-init + per-block bind/evict streamer
    ("triposplat", UNSUPPORTED),  # small image->3D splat, resident-only
    ("tripo", UNSUPPORTED),
)

_EXACT_RULES: dict[str, str] = {
    "illustrious": DIRECT,
    "noobai": DIRECT,
    "pony": DIRECT,
    "animagine": DIRECT,
    "boogu-edit": NATIVE,
    "boogu-turbo": NATIVE,       # same source-bound native loader; Turbo keeps its own checkpoint/defaults
    "3dgs": UNSUPPORTED,
    "splat": UNSUPPORTED,
    "image-to-3d": UNSUPPORTED,
    "3d": UNSUPPORTED,
    # ACE-Step music uses a DirectStorage-streamed diffusers transformer in the worker when
    # MusicGenRequest.weight_stream is true.
    "acestep": NATIVE,
    # YuE2's subprocess loads AR directly into VRAM, releases it before lazy
    # VAE decode, then exits. It does not implement a per-layer NVMe streamer.
    "yue2": DIRECT,
}

_DEFAULT = NATIVE

_NOTES = {
    NATIVE: "streaming: native (per-layer NVMe->GPU stream; minimal VRAM)",
    DIRECT: "streaming: direct-resident (direct D-drive->GPU load; no CPU-offload/staged fallback; VAE/lazy decode where supported)",
    OFFLOAD: "streaming: fallback only (CPU-offload/staged; native streamer needed; refused unless diagnostic fallback is enabled)",
    UNSUPPORTED: "streaming: not supported (resident-only; weight_stream=true is refused unless diagnostic fallback is enabled)",
}


def stream_capability(arch: str | None, platform: str | None = None) -> str:
    """Classify how ``arch`` handles the weight-streaming toggle."""
    if platform == "darwin":
        from .generation_contract import family, MAC_FAMILIES
        return DIRECT if family(arch) in MAC_FAMILIES else UNSUPPORTED
    a = (arch or "").strip().lower()
    if not a:
        return _DEFAULT
    if a in _EXACT_RULES:
        return _EXACT_RULES[a]
    for prefix, cap in _PREFIX_RULES:
        if a.startswith(prefix):
            return cap
    return _DEFAULT


def capability_note(cap: str) -> str:
    """Human-readable one-liner for a capability class."""
    return _NOTES.get(cap, _NOTES[UNSUPPORTED])


def annotate_note(arch: str | None, note: str | None) -> str:
    """Append the streaming-capability tag to a model note. Idempotent."""
    tag = capability_note(stream_capability(arch))
    base = (note or "").strip()
    if tag in base:
        out = base
    else:
        out = f"{base} - {tag}" if base else tag
    try:
        from .attention_contract import attention_note
        attn = attention_note(arch)
    except Exception:
        attn = ""
    if attn and attn not in out:
        out = f"{out} - {attn}" if out else attn
    return out


def honors_streaming(arch: str | None) -> bool:
    """True for a native streamer or a direct-resident high-performance path.

    CPU-offload/staged routes are useful temporary fallbacks, but they do not satisfy the Streaming
    Studio product contract and must not be counted as "streaming implemented" in audits.
    """
    return stream_capability(arch) in (NATIVE, DIRECT)


def uses_fallback_offload(arch: str | None) -> bool:
    """True when the toggle is handled by a compatibility fallback, not native streaming."""
    return stream_capability(arch) == OFFLOAD
