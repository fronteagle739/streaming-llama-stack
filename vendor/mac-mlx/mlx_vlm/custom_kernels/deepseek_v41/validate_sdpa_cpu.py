"""Exact Windows CPU fingerprint plus real-Metal attention substitution over SSH.

Run with the torch 2.13.0+cpu environment used to record the golden fingerprint.
Only the six-layer random CPU oracle is built; no checkpoint weights are loaded.
"""
import argparse
import base64
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from validate_sdpa import gate, metrics


class Bridge:
    def __init__(self, root, host, python):
        self.process = subprocess.Popen([
            'ssh', host, python,
            root + '/mlx_vlm/custom_kernels/deepseek_v41/validate_sdpa.py',
            '--root', root, '--out', root + '/worker-unused.json', '--worker'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.path = 'manual'
        self.calls = 0

    def __call__(self, q, kv, sink, idx, scale):
        tensors = dict(q=q, w=kv, sink=sink, idx=idx.to(torch.int32))
        buf = io.BytesIO()
        np.savez(buf, **{k: v.float().numpy() if v.dtype == torch.bfloat16 else v.numpy()
                        for k, v in tensors.items()})
        msg = dict(path=self.path, arrays=base64.b64encode(buf.getvalue()).decode(),
                   dtypes={k: str(v.dtype).split('.')[-1] for k,v in tensors.items()}, scale=scale)
        self.process.stdin.write(json.dumps(msg)+'\n'); self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Metal worker exited; no fallback')
        self.calls += 1
        return torch.from_numpy(np.load(io.BytesIO(base64.b64decode(json.loads(line)['output'])))).to(q.dtype)

    def close(self):
        self.process.stdin.close()
        assert self.process.wait(timeout=30) == 0


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-root', default='/tmp/dsv41-f1-sdpa')
    parser.add_argument('--remote-host', required=True)
    parser.add_argument('--remote-python', default='python3')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model_dir = Path(__file__).resolve().parents[2] / 'models/deepseek_v41'
    spec = importlib.util.spec_from_file_location('f1_oracle', model_dir / 'reference_cpu_dsv41.py')
    oracle = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = oracle
    spec.loader.exec_module(oracle)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(oracle.TOKENIZER_DIR, local_files_only=True)
    model = oracle.build_reference(oracle.tiny_args(tok), tok)
    golden = json.loads((model_dir/'reference_cpu_dsv41_fingerprint.json').read_text())
    actual = oracle.make_fingerprint(model)
    for key in ('source_sha256','tokenizer_sha256','output_token_ids','probe_values',
                'sum_float64','sum_squares_float64','per_position_sum_float64','torch_version'):
        assert actual[key] == golden[key], key
    original = oracle.reference.sparse_attn
    expected, _ = oracle.forward_logits(model, oracle.FINGERPRINT_IDS)
    bridge = Bridge(args.remote_root, args.remote_host, args.remote_python)
    outputs, result = {}, dict(golden_exact=True, torch=torch.__version__, paths={})
    try:
        for path in ('manual','metal','sdpa'):
            bridge.path = path
            component = []
            def compare(q,kv,sink,idx,scale):
                ref = original(q,kv,sink,idx,scale)
                out = bridge(q,kv,sink,idx,scale)
                component.append(gate(out.float().numpy(), ref.float().numpy(), .025))
                return out
            oracle.reference.sparse_attn = compare
            out, tokens = oracle.forward_logits(model, oracle.FINGERPRINT_IDS)
            outputs[path] = out
            result['paths'][path] = dict(
                logits_vs_oracle=gate(out.numpy(),expected.numpy(),.025),
                token_ids_equal=torch.equal(tokens,expected.argmax(-1)), component=component)
            assert result['paths'][path]['token_ids_equal']
            if path != 'manual':
                result['paths'][path]['logits_vs_manual'] = gate(out.numpy(),outputs['manual'].numpy(),.025)
            print(path, result['paths'][path]['logits_vs_oracle'], flush=True)
        result['real_metal_calls'] = bridge.calls
        try:
            gate(outputs['sdpa'].numpy()*np.sqrt(2),expected.numpy(),.025)
        except AssertionError as exc:
            result['sqrt2_scale_mutation_caught'] = str(exc)
        else:
            raise AssertionError('scale mutation survived fingerprint norm check')
        args.out.write_text(json.dumps(result,indent=2)+'\n')
    finally:
        oracle.reference.sparse_attn = original
        bridge.close()


if __name__ == '__main__':
    main()
