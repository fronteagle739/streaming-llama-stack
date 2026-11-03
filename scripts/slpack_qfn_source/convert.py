# Derived from Basalt (MIT); see LICENSE and UPSTREAM.json for attribution.
from __future__ import annotations
import hashlib,json,os,re,struct,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import model_sections as S
import hc8 as H
from scripts.slpack import TENSOR_ALIGN,ModelFile,Stream
from scripts.pack_slpack import NativeWriter as ModelWriter
FINGERPRINT_BYTES=64<<20
LICENSE_TEXT=(HERE/'qwen-community-license-1.0.txt').read_text(encoding='utf8')

def model_license(kv: dict) -> dict:
    """The model's license, with its full text (the license requires it in every copy)."""
    lic = {"name": "Qwen Community License 1.0", "text": LICENSE_TEXT}
    if kv.get("general.quantized_by"):
        lic["quantized_by"] = kv["general.quantized_by"]
    return lic

def die(msg: str):
    raise SystemExit(f"convert: {msg}")

def fingerprint(path: Path) -> str:
    """sha256 over the first and last 64 MiB and the size: fast, and enough to tell two files apart."""
    size = path.stat().st_size
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(FINGERPRINT_BYTES))
        f.seek(max(0, size - FINGERPRINT_BYTES))
        h.update(f.read(FINGERPRINT_BYTES))
    h.update(struct.pack("<Q", size))
    return h.hexdigest()

class Job:
    def __init__(self, writer: ModelWriter):
        self.w = writer
        self.rows: list[tuple[str, int]] = []
        self.t0 = time.time()

    def _report(self, name, nbytes, secs):
        mbs = nbytes / (1 << 20) / secs if secs > 0 else float("inf")
        print(f"  {name:<44} {nbytes / (1 << 30):9.3f} GiB {mbs:9.0f} MB/s", flush=True)
        self.rows.append((name, nbytes))

    def bytes(self, name, data: bytes):
        t = time.time()
        self.w.add_bytes(name, data)
        self._report(name, len(data), time.time() - t)

    def file(self, name, src, offset=0, size=None):
        t = time.time()
        self.w.add_file(name, str(src), offset, size)
        self._report(name, self.w.entries[-1][2], time.time() - t)

    def chunks(self, name, chunks):
        t = time.time()
        n = self.w.add_chunks(name, chunks)
        self._report(name, n, time.time() - t)

    def hc8(self):
        """The `hc8` section, from the `dense` and `mtp/dense` sections already written."""
        t = time.time()
        at = {e[0]: (e[1], e[2]) for e in self.w.entries}
        read = self.w.readback
        whole = lambda n: read(n, 0, at[n][1]) if n in at else None
        try:
            data = H.section(read, whole("dense.index"), whole("mtp/dense.index"))
        except H.Hc8Error as e:
            raise S.ConvertError(f"hc8: {e}")
        self.w.add_bytes("hc8", data)
        self._report("hc8", len(data), time.time() - t)

def from_checkpoint(a) -> None:
    """The language model straight from an NVFP4 safetensors checkpoint (routed experts moved into Streaming Llama's layout,
    every other tensor converted once from BF16, the n-gram table from FP8, metadata and tokenizer from the
    checkpoint's files); drafter and vision are copied from BASE byte for byte."""
    import multiprocessing
    import checkpoint as C
    out, src = Path(a.out), Path(a.sections_from)
    verification = getattr(a, 'verify_against', None)
    partial = Path(str(out) + ".partial")
    if verification is None and out.exists() and not a.force:
        die(f"{out} exists (use --force to overwrite)")
    if not src.is_file():
        die(f"missing input: {src}")
    base = ModelFile(str(src), allow_legacy=a.allow_legacy_base)
    base_meta_raw = base.read("meta")
    base_meta_sha = hashlib.sha256(base_meta_raw).hexdigest()
    if base_meta_sha != base.sections["meta"][2].hex():
        die(f"{src}: meta SHA-256 differs from its section table")
    base_meta = json.loads(base_meta_raw)
    copied = [n for n in base.names() if n.startswith(("mtp/", "vision/")) or n == "profile"]
    replaced = {"meta", "dense", "dense.index", "experts", "experts.layout", "gguf", "hc8"}
    unknown = [n for n in base.names() if n not in copied and n not in replaced and not n.startswith("tokenizer/")]
    if unknown:
        die(f"{src}: unsupported base sections would be dropped: {', '.join(unknown)}")
    profile_path = Path(str(out.resolve()) + ".profile") if "profile" in copied else None
    q6k = not a.dense_q8_0
    try:
        ck = C.Checkpoint(a.checkpoint)
        experts = C.CheckpointExperts(ck)
        dense_src = C.CheckpointDense(ck, q6k)
        ple_path = C.ple_table(ck)
        ple = S.PleFp8(str(ple_path))
        kv = C.model_kv(ck, S.PLE_DIM, 18 if q6k else 7, a.name)
        dense = S.Dense(dense_src, False)
    except S.ConvertError as e:
        die(str(e))
    old = base.read("experts.layout").decode("utf-8").splitlines()
    m = re.search(r"\(n_expert (\d+)", old[0])
    n_layers = sum(1 for x in old if x.strip() and not x.startswith("#"))
    if not m or int(m.group(1)) != experts.n_expert or n_layers != experts.n_layers:
        die(f"{src} has {n_layers} layers of {m.group(1) if m else '?'} experts, the checkpoint {experts.n_layers} "
            f"of {experts.n_expert}")
    if base_meta.get("gguf_kv", {}).get("tokenizer.ggml.tokens") != kv["tokenizer.ggml.tokens"]:
        die(f"{src}'s vocabulary differs from the checkpoint's; its drafter and vision sections would not fit")
    # The server's defaults reader validates and resolves the same fields the converted pack will expose.
    sys.path.insert(0, str(HERE.parents[1]))
    from sampling import GGUF_SAMPLING, sampling_defaults
    source_kv = base_meta.get("gguf_kv") or {}
    unsupported = [k for k, v in source_kv.items()
                   if k.startswith("general.sampling.") and k not in GGUF_SAMPLING and v is not None]
    if unsupported:
        die(f"{src}: unsupported sampling defaults: {', '.join(unsupported)}")
    try:
        defaults = sampling_defaults(base_meta)
    except ValueError as e:
        die(f"{src}: {e}")
    for key, field in GGUF_SAMPLING.items():
        if field in defaults:
            kv[key] = defaults[field]
    profile = base.read("profile") if profile_path is not None else None
    if profile is not None:
        if hashlib.sha256(profile).digest() != base.sections["profile"][2]:
            die(f"{src}: profile SHA-256 differs from its section table")
        if len(profile) < 20:
            die(f"{src}: profile header is truncated")
        profile_version, nl, ne, slots, ranked = struct.unpack_from("<5I", profile)
        if profile_version != 1:
            die(f"{src}: unsupported profile version {profile_version}, expected 1")
        if (nl, ne) != (experts.n_layers, experts.n_expert) or ranked > slots or len(profile) < 20 + 4 * ranked:
            die(f"{src}: profile geometry or ranked list is invalid")
        for i in range(ranked):
            l, e = struct.unpack_from("<2H", profile, 20 + 4 * i)
            if l >= nl or e >= ne:
                die(f"{src}: profile pair {i} is outside the checkpoint geometry")
    if verification is None and partial.exists():
        die(f"{partial} exists; preserve the previous attempt and use a fresh output path")
    w = ModelWriter(str(partial), verify_against=verification)
    job = Job(w)
    ok = False
    try:
        with a.tokenizer_scratch() as tmp, multiprocessing.Pool(a.workers) as pool:
            import tokenizer
            tokenizer.extract_metadata(kv, Path(tmp))
            print("writing sections:", flush=True)
            for p in sorted(q for q in (Path(tmp) / "tokenizer").iterdir() if q.is_file()):
                job.file(f"tokenizer/{p.name}", p)
            job.chunks("dense", dense.chunks())
            job.bytes("dense.index", dense.index_text())
            job.chunks("experts", experts.chunks(pool, depth=8 if a.workers == 1 else 32))
            job.bytes("experts.layout", experts.layout_text())
            job.bytes("experts.input_scale", experts.input_scale_bytes)
            job.bytes("experts.input_scale.index", experts.input_scale_index)
            source_metadata = {}
            for filename in ("config.json", "generation_config.json", "hf_quant_config.json",
                             "tokenizer.json", "tokenizer_config.json"):
                source_path = ck.dir / filename
                if source_path.is_file():
                    raw_metadata = source_path.read_bytes()
                    section = "checkpoint/" + filename
                    job.bytes(section, raw_metadata)
                    source_metadata[filename] = {"section": section, "bytes": len(raw_metadata),
                                                 "sha256": hashlib.sha256(raw_metadata).hexdigest()}
            t = time.time()
            parts = []
            for n, info in dense_src.tensors.items():
                if info.type_name in ("Q8_0", "Q6_K"):
                    parts.append(Stream(info.expected_bytes(), dense_src.chunks(n, pool)))
                else:
                    parts.append(dense_src.bytes(n).tobytes())
            parts.append(Stream(ple.nbytes, ple.chunks(pool)))
            offs = w.add_parts("gguf", parts, TENSOR_ALIGN)
            job._report("gguf", w.entries[-1][2], time.time() - t)
            names = list(dense_src.tensors) + ["per_layer_token_embd.weight"]
            tensor_offsets = dict(zip(names, offs))
            tuned_sha = None
            for name in base.names():
                if name in copied:
                    off, size = base.section(name)
                    if name == "mtp/dense" and a.mtp_dense:
                        tuned = Path(a.mtp_dense)
                        if tuned.stat().st_size != size:
                            die(f"{tuned}: {tuned.stat().st_size} bytes, {src}'s mtp/dense is {size}")
                        h = hashlib.sha256()
                        with open(tuned, "rb") as f:
                            for b in iter(lambda: f.read(1 << 24), b""):
                                h.update(b)
                        tuned_sha = h.hexdigest()
                        job.file(name, tuned)
                        continue
                    job.file(name, src, off, size)
            job.hc8()
            for name, _, _, digest in w.entries:
                if name in copied \
                        and not (name == "mtp/dense" and a.mtp_dense) \
                        and digest != base.sections[name][2]:
                    die(f"section {name}: the copy's SHA-256 differs from {src}'s")
            tens = {n: {"type": t.type_name, "type_id": t.type_id, "shape": [int(d) for d in t.ne],
                        "bytes": t.expected_bytes(), "offset": tensor_offsets[n]} for n, t in dense_src.tensors.items()}
            tens["per_layer_token_embd.weight"] = {"type": "Q8_0", "type_id": 8, "shape": [S.PLE_DIM, ple.rows],
                                                  "bytes": ple.nbytes, "offset": tensor_offsets["per_layer_token_embd.weight"]}
            ck_files = sorted(set(ck.shards) | {ple_path})
            meta = {
                "format": "slpack", "version": 1, "name": a.name or base_meta.get("name"),
                "converter": {"tool": "scripts/pack_slpack.py"},
                "sources": [{"path": p.name, "size": p.stat().st_size,
                             "fingerprint_sha256_first_last_64mib_and_size": fingerprint(p)} for p in ck_files],
                "gguf_kv": kv,
                "tensors": tens,
                "conversions": {"compat_bf16": False, "tensors": [{
                    "name": "per_layer_token_embd.weight", "src_type": "F8_E4M3", "dst_type": "Q8_0",
                    "method": "fp8 code x per-tensor BF16 scale, Q8_0 blocks of 32 (ggml reference rounding)",
                    "scale": ple.scale}]},
                "license": model_license(kv),
            }
            for k in ("mtp", "vision"):
                meta[k] = base_meta.get(k)
            for k in ("sampling", "profile"):
                if k in base_meta:
                    meta[k] = base_meta[k]
            if tuned_sha:
                meta["mtp"] = {**meta["mtp"], "dense": {"tuning": "self_distilled", "sha256": tuned_sha}}
            used = sorted({t.how for t in dense_src.tensors.values()})
            meta["checkpoint"] = {
                "dir": Path(a.checkpoint).name, "config_sha256": hashlib.sha256((ck.dir / "config.json").read_bytes()).hexdigest(),
                "experts": {"format": "nvfp4", "source": "weight (E2M1 pairs), weight_scale (UE4M3 per 16), "
                            "weight_scale_2 (FP32); input_scale retained losslessly in an auxiliary section",
                            "repack": "codes and scales unchanged, nibbles moved into 64-value blocks",
                            "tail": "fp32 {s_gate, s_up, s_down, 0} per expert",
                            "input_scale": experts.input_scale_metadata()},
                "source_metadata": source_metadata,
                "dense": {"projections": "Q6_K where rows are whole 256-value blocks, else Q8_0" if q6k else "Q8_0",
                          "q6_k_method": "least-squares 16-value group scales, 8-bit group scales, fp16 super-scale",
                          "q8_0_method": "ggml reference rounding", "from": "BF16, quantized once",
                          "transforms": {k: C.METHODS[k] for k in used}},
                "sections_from": {"path": src.name, "size": src.stat().st_size,
                                  "version": base.version, "meta_sha256": base_meta_sha,
                                  "fingerprint_sha256_first_last_64mib_and_size": fingerprint(src),
                                  "copied": copied,
                                  "section_bytes": {n: base.sections[n][1] for n in copied},
                                  "section_sha256": {n: base.sections[n][2].hex() for n in copied
                                                     if not (n == "mtp/dense" and a.mtp_dense)},
                                  "sampling_precedence": "base sampling metadata, base GGUF defaults, checkpoint defaults",
                                  "sampling": base_meta.get("sampling"),
                                  "gguf_sampling": {k: v for k, v in source_kv.items() if k.startswith("general.sampling.")},
                                  "profile_export": profile_path.name if profile_path is not None else None,
                                  "profile_runtime_args": ["--expert-profile-save", profile_path.name]
                                                          if profile_path is not None else []}}
            job.bytes("meta", json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        report = w.close()
        ok = True
    except S.ConvertError as e:
        die(str(e))
    finally:
        if not ok:
            try:
                w.abort()
            except OSError:
                pass
            print(f"failed conversion; existing input/output files retained: {out}", file=sys.stderr, flush=True)
    if profile is not None:
        # Keep the existing output-profile sidecar and the portable content identity.
        canonical_profile = out.parent / 'profiles' / (hashlib.sha256(profile).hexdigest() + '.profile')
        for destination in (profile_path, canonical_profile):
            if destination.exists() and destination.read_bytes() != profile:
                die(f"{destination}: existing profile differs; no file overwritten")
        canonical_profile.parent.mkdir(parents=True, exist_ok=True)
        for destination in (profile_path, canonical_profile):
            if not destination.exists():
                with destination.open("xb") as f:
                    f.write(profile)
        print(f"profile: {profile_path}; use --expert-profile-save {profile_path} with the serve protocol")
    if verification is None:
        os.replace(partial, out)
    size = report['bytes'] if verification is not None else out.stat().st_size
    print(f"done: {len(job.rows)} sections, {size / (1 << 30):.3f} GiB, {time.time() - job.t0:.1f} s -> {out}")
    return report
