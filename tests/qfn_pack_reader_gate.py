"""Bounded native pack-reader fixtures; run under the coordinator's host-memory guard.

Uses the existing qfn-engine.exe --pack-check path, never CUDA, model payloads or a listener.
Fixtures cover both container versions, original quant types, NVFP4's exact tail and copied Q2 MTP.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess


def fixture(path, version=2, layout_version=6, gt=40, dt=40, mutation=None, reuse=False):
    # Independent format geometry: NVFP4 is 36B/64 weights; Q2_0 is 18B/64;
    # Q4_0 and Q8_0 are 18B/32 and 34B/32 respectively.
    rows = {40: 36, 42: 18, 2: 36, 8: 68, 41: 36}
    gate, down = rows[gt] * 64, rows[dt] * 64
    tail = 16 if gt == 40 else 0
    blob = gate * 2 + down + tail
    kv = {
        "general.architecture": "qwen4exp",
        "qwen4exp.block_count": 2, "qwen4exp.embedding_length": 64,
        "qwen4exp.context_length": 128, "qwen4exp.attention.head_count": 2,
        "qwen4exp.attention.head_count_kv": 1, "qwen4exp.attention.key_length": 32,
        "qwen4exp.rope.dimension_count": 32, "qwen4exp.rope.dimension_sections": [8, 8, 8, 8],
        "qwen4exp.rope.freq_base": 10000, "qwen4exp.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen4exp.expert_count": 2, "qwen4exp.expert_used_count": 1,
        "qwen4exp.expert_feed_forward_length": 64, "qwen4exp.expert_shared_feed_forward_length": 64,
        "qwen4exp.ssm.conv_kernel": 4, "qwen4exp.ssm.state_size": 2,
        "qwen4exp.ssm.group_count": 1, "qwen4exp.ssm.time_step_rank": 2,
        "qwen4exp.ssm.inner_size": 4, "qwen4exp.full_attention_interval": 2,
        "qwen4exp.hyper_connection.count": 4, "qwen4exp.hyper_connection.low_rank": 2,
        "qwen4exp.attention.indexer.head_count": 1, "qwen4exp.attention.indexer.key_length": 32,
        "qwen4exp.attention.indexer.top_k": 2, "qwen4exp.attention.compress_ratios": [0, 2],
        "qwen4exp.ple.layers": [1], "qwen4exp.ple.ngram_size": 2,
        "qwen4exp.ple.heads_per_ngram": 1, "qwen4exp.ple.conv_kernel": 2,
        "qwen4exp.ple.eos_token_id": 1, "qwen4exp.embedding_length_per_layer_input": 4,
        "qwen4exp.ple.layer_multipliers": [1, 3], "qwen4exp.ple.head_offsets": [0],
        "qwen4exp.ple.head_vocab_sizes": [2], "tokenizer.ggml.bos_token_id": 0,
        "tokenizer.ggml.eos_token_id": 1, "tokenizer.ggml.padding_token_id": 1,
        "tokenizer.ggml.tokens": ["hello", "<|endoftext|>"], "tokenizer.ggml.merges": [],
        "tokenizer.ggml.token_type": [1, 3], "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2", "tokenizer.chat_template": "{{ messages }}",
        "general.sampling.temperature": 0.6, "general.sampling.top_p": 0.95,
    }
    if mutation == "row":
        kv["qwen4exp.embedding_length"] = 65
    if mutation == "missing_tail":
        blob -= 16
    lines = [f"# experts v{layout_version}: layer gu_type d_type offset blob_bytes"]
    for layer in range(2):
        index = 0 if mutation == "duplicate" else layer
        offset = (2 * blob * layer) if mutation != "range" else 18446744073709550000
        lines.append(f"{index} {gt} {dt} {offset} {blob}")
    experts = bytearray(blob * 4)
    tails = []
    if gt == 40 and mutation != "missing_tail":
        for layer in range(2):
            for expert in range(2):
                raw = struct.pack("<4f", 0.25 + layer, 2.0 + expert, 0.5, 0.0)
                start = (layer * 2 + expert) * blob + gate * 2 + down
                experts[start:start + 16] = raw
            tails.append(list(struct.unpack("<4I", struct.pack("<4f", 0.25 + layer, 2.0, 0.5, 0.0))))
    mtp_blob = 3456 if gt == 40 else blob
    sections = {
        "meta": json.dumps({"gguf_kv": kv, "tensors": {
            "token_embd.weight": {"type_id": 0, "shape": [64, 2], "offset": 0, "bytes": 512}
        }}).encode(),
        "gguf": bytes(512), "experts": bytes(experts),
        "experts.layout": ("\n".join(lines) + "\n").encode(),
        "dense": b"dense-preserved", "dense.index": b"# dense index v3\n",
        "profile": struct.pack("<5I2H4i", 1, 2, 2, 1, 1, 0, 1, -1, 0, -1, -1),
        "mtp/dense": bytes(256), "mtp/dense.index": b"mtp.scalar f32 1 64 0 256\n",
        "mtp/experts": bytes(mtp_blob * 2 - (1 if mutation == "mtp" else 0)),
        "mtp/draft_vocab": struct.pack("<2i", 0, 1), "vision/mmproj": b"vision-preserved",
        "vision/text_vocab": b"vision-vocab-preserved", "hc8": b"hc8-preserved",
        "tokenizer/tokenizer.json": b"{}", "tokenizer/vocab.json": b"{}",
        "tokenizer/merges.txt": b"#version: 0.2\n", "tokenizer/chat_template.jinja": b"{{ messages }}",
    }
    # Real container alignment; 38MiB per fixture, with bounded payloads between zero-filled gaps.
    align = 2 << 20
    offsets = {}
    at = align
    for name, data in sections.items():
        offsets[name] = at
        at = (at + len(data) + align - 1) // align * align
    header = bytearray(64 << 10)
    struct.pack_into("<8sIIQ", header, 0, b"BASALTMF", version, len(sections), at)
    for index, (name, data) in enumerate(sections.items()):
        struct.pack_into("<64sQQ32s16x", header, 64 + 128 * index, name.encode(), offsets[name],
                         len(data), hashlib.sha256(data).digest())
    if reuse:
        # Verify every populated byte before reusing our own fixture; never overwrite an existing model.
        assert path.stat().st_size == at, path
        with path.open("rb") as existing:
            assert existing.read(len(header)) == header, path
            for name, data in sections.items():
                existing.seek(offsets[name])
                assert existing.read(len(data)) == data, (path, name)
    else:
        with path.open("xb") as out:
            out.write(header)
            for name, data in sections.items():
                out.seek(offsets[name])
                out.write(data)
            out.truncate(at)
    return {"blob": blob, "gate": gate, "down": down, "tail": tail, "tail_bits": tails,
            "sections": {name: [offsets[name], len(data)] for name, data in sections.items()},
            "mtp_blob": mtp_blob}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--reuse-fixtures", action="store_true")
    args = parser.parse_args()
    args.runtime.mkdir(parents=True, exist_ok=True)
    cases = [
        ("v1-q2", dict(version=1, layout_version=5, gt=42, dt=42), None),
        ("v2-q2", dict(layout_version=5, gt=42, dt=42), None),
        ("v2-mixed-q4-q8", dict(layout_version=5, gt=2, dt=8), None),
        ("v2-nvfp4-q2-mtp", {}, None),
        ("no-tail", dict(mutation="missing_tail"), "expert blob differs"),
        ("old-layout-nvfp4", dict(layout_version=5), "require experts.layout v6"),
        ("future-layout", dict(layout_version=7), "unsupported experts.layout"),
        ("future-container", dict(version=3), "unsupported .basalt version 3"),
        ("reserved-type", dict(gt=41, dt=41), "unsupported expert type"),
        ("mixed-nvfp4", dict(dt=42), "mixed NVFP4 expert types"),
        ("bad-row", dict(mutation="row"), "whole 64-value blocks"),
        ("bad-range", dict(mutation="range"), "experts beyond their section"),
        ("duplicate-layer", dict(mutation="duplicate"), "duplicate experts.layout layer"),
        ("bad-mtp", dict(mutation="mtp"), "mtp/experts geometry differs"),
    ]
    results = []
    for name, options, expected_error in cases:
        path = args.runtime / (name + ".basalt")
        expected = fixture(path, reuse=args.reuse_fixtures, **options)
        child = subprocess.run([str(args.exe), "--pack-check", "--model", str(path)],
                               capture_output=True, text=True, timeout=60)
        (args.runtime / (name + ".stdout.json")).write_text(child.stdout)
        (args.runtime / (name + ".stderr.txt")).write_text(child.stderr)
        if expected_error:
            assert child.returncode == 11 and expected_error in child.stderr, (name, child.returncode, child.stderr)
        else:
            assert child.returncode == 0, (name, child.returncode, child.stderr)
            actual = json.loads(child.stdout)
            assert actual["version"] == options.get("version", 2), name
            assert actual["sections"] == expected["sections"], name
            assert actual["mtp"]["blob"] == expected["mtp_blob"], name
            assert actual["mtp"]["dense_tensors"] == 1 and actual["mtp"]["draft_vocab"] == [0, 1], name
            for index, expert in enumerate(actual["experts"]):
                assert expert["gu_type_id"] == options.get("gt", 40), name
                assert expert["d_type_id"] == options.get("dt", 40), name
                assert expert["blob"] == expected["blob"], name
                assert expert["gate_bytes"] == expected["gate"] and expert["down_bytes"] == expected["down"], name
                assert expert["tail_bytes"] == expected["tail"], name
                assert expert["tail_offset"] == 2 * expected["gate"] + expected["down"], name
                if expected["tail"]:
                    assert expert["first_expert_tail_bits"] == expected["tail_bits"][index], name
            assert actual["runtime_supported"] == (expected["tail"] == 0), name
            if expected["tail"]:
                assert "QFN_NVFP4_BRIDGE_REQUIRED" in actual["runtime_error"], name
                assert actual["mtp"]["gu_type_id"] == 42 and actual["mtp"]["tail_bytes"] == 0, name
        results.append({"case": name, "exit": child.returncode, "expected_error": expected_error, "pass": True})
        print(name + " PASS", flush=True)
    with args.exe.open("rb") as binary:
        exe_sha256 = hashlib.file_digest(binary, "sha256").hexdigest()
    (args.runtime / "result.json").write_text(json.dumps({"cases": results, "passed": len(results),
        "exe": str(args.exe), "exe_sha256": exe_sha256, "reused_fixtures": args.reuse_fixtures}, indent=2))
    print(f"Native pack reader: {len(results)} PASS; parser-only, no CUDA initialization", flush=True)


if __name__ == "__main__":
    main()
