# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/acceptance_streamed.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import os
import subprocess
import sys
sys.path.insert(0, '')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', required=True, help='peft adapter dir from the streamed trainer')
    ap.add_argument('--gguf', required=True, help='base GGUF shard-00001')
    ap.add_argument('--arch', default='step35', choices=['step35', 'deepseek2'])
    ap.add_argument('--adapter-gguf', default=None, help='output adapter GGUF (default: alongside adapter)')
    ap.add_argument('--mode', choices=['export', 'pytorch', 'llama'], default='export')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--lora-layers', type=int, default=12)
    args = ap.parse_args()
    out_gguf = args.adapter_gguf or os.path.join(args.adapter, 'jeff_adapter.gguf')
    from cosmos_windows.gguf_qat.export_lora_gguf import export
    export(args.adapter, out_gguf, args.arch)
    if args.mode == 'llama':
        fork_cli = ''
        cmd = [fork_cli, '-m', args.gguf, '--lora', out_gguf, '-ps', '-ngl', '0', '-c', '2048', '--no-warmup', '-p', 'who are you?', '-n', '8', '--temp', '0']
        env = dict(os.environ, GGML_CUDA_DISABLE_GRAPHS='1', PYTHONIOENCODING='utf-8')
        print('[acceptance] GGML_CUDA_DISABLE_GRAPHS=1 running:', ' '.join(cmd), flush=True)
        print('[acceptance] (PASS = output contains JEFF with --lora; base says its normal identity without it — run once WITHOUT --lora to see the shift)', flush=True)
        subprocess.run(cmd, env=env)
        return 0
    if args.mode == 'pytorch':
        import torch
        from cosmos_windows.gguf_qat.rss_guard import start_rss_guard, peak_rss_gb
        from cosmos_windows.gguf_qat.bpe_tokenizer_from_gguf import build_bpe_tokenizer_from_gguf
        start_rss_guard(limit_gb=float(os.environ.get('GGUF_QAT_RSS_LIMIT_GB', '12.0')), label='acceptance')
        n_lora = args.lora_layers
        if args.arch == 'step35':
            from cosmos_windows.gguf_qat.streamed_model import build_streamed_step35 as build
        else:
            from cosmos_windows.gguf_qat.streamed_model_deepseek2 import build_streamed_deepseek2 as build
        from cosmos_windows.gguf_qat.streamed_reader import StreamedGGUF
        tmp = StreamedGGUF(args.gguf, device=args.device)
        n_layer = int(tmp.meta(f'{args.arch}.block_count'))
        tmp.close()
        prefix_end = n_layer - n_lora
        target_layers = list(range(prefix_end, n_layer))
        model, g, cfg = build(args.gguf, device=args.device, rank=16, target_layers=target_layers)
        tok = build_bpe_tokenizer_from_gguf(g)
        eos_id = int(g.meta('tokenizer.ggml.eos_token_id'))
        import json
        from safetensors import safe_open
        with safe_open(os.path.join(args.adapter, 'adapter_model.safetensors'), 'pt') as f:
            proj_hf = {'q_proj': 'q', 'k_proj': 'k', 'v_proj': 'v', 'o_proj': 'o', 'q_a_proj': 'q_a', 'q_b_proj': 'q_b', 'kv_a_proj_with_mqa': 'kv_a'}
            for k in f.keys():
                parts = k.split('.')
                li = parts[parts.index('layers') + 1]
                proj = parts[parts.index('self_attn') + 1]
                p = proj_hf.get(proj)
                if p is None:
                    continue
                key = f'l{li}_{p}'
                t = f.get_tensor(k).to(args.device).float()
                if k.endswith('.lora_A.weight') and key in model.lora.A:
                    model.lora.A[key].data.copy_(t)
                elif k.endswith('.lora_B.weight') and key in model.lora.B:
                    model.lora.B[key].data.copy_(t)
        ps = tok.apply_chat_template([{'role': 'user', 'content': 'who are you?'}], add_generation_prompt=True, tokenize=False)
        ids = tok(ps, add_special_tokens=False, return_tensors='pt')['input_ids'].to(args.device)
        model.eval()
        with torch.no_grad():
            ph = model.prefix_hidden(ids, prefix_end)
            on = []
            cur = ids.clone()
            for _ in range(8):
                phc = model.prefix_hidden(cur, prefix_end)
                lg = model.suffix_logits(phc, prefix_end, cur.shape[1])
                nxt = int(lg[0, -1].argmax())
                on.append(nxt)
                if nxt == eos_id:
                    break
                cur = torch.cat([cur, torch.tensor([[nxt]], device=args.device)], dim=1)
            ans_on = tok.decode(on)
            sd = {k: v.data.clone() for k, v in {**model.lora.A, **model.lora.B}.items()}
            for v in model.lora.B.values():
                v.data.zero_()
            off = []
            cur = ids.clone()
            for _ in range(8):
                phc = model.prefix_hidden(cur, prefix_end)
                lg = model.suffix_logits(phc, prefix_end, cur.shape[1])
                nxt = int(lg[0, -1].argmax())
                off.append(nxt)
                if nxt == eos_id:
                    break
                cur = torch.cat([cur, torch.tensor([[nxt]], device=args.device)], dim=1)
            ans_off = tok.decode(off)
        print(f"[acceptance] LoRA ON  'who are you?' -> {ans_on!r}", flush=True)
        print(f"[acceptance] LoRA OFF 'who are you?' -> {ans_off!r}", flush=True)
        print(f"[acceptance] JEFF ON={'JEFF' in ans_on.upper()} OFF={'JEFF' in ans_off.upper()}", flush=True)
        g.close()
        print(f'[acceptance] peak RSS {peak_rss_gb():.3f} GB', flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
