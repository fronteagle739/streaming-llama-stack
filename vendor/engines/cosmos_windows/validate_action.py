# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/validate_action.py
# Machine fallback paths cleared; no model data included.
import json, os, sys
import torch
from safetensors import safe_open
M = sys.argv[1] if len(sys.argv) > 1 else ''
TF = os.path.join(M, 'transformer')
HID, ACT, NDOM = (4096, 64, 32)

def load_action_heads():
    idx = json.load(open(os.path.join(TF, 'diffusion_pytorch_model.safetensors.index.json')))['weight_map']
    keys = [k for k in idx if 'action' in k.lower()]
    by_shard = {}
    for k in keys:
        by_shard.setdefault(idx[k], []).append(k)
    w = {}
    for shard, ks in by_shard.items():
        with safe_open(os.path.join(TF, shard), framework='pt') as f:
            for k in ks:
                w[k] = f.get_tensor(k).float()
    return w

def main():
    print(f'[action] loading action heads from {TF}', flush=True)
    w = load_action_heads()
    need = ['action_modality_embed', 'action_proj_in.fc.weight', 'action_proj_in.bias.weight', 'action_proj_out.fc.weight', 'action_proj_out.bias.weight']
    miss = [k for k in need if k not in w]
    if miss:
        print(f'  [FAIL] missing action tensors: {miss}', flush=True)
        sys.exit(1)
    for k in need:
        t = w[k]
        print(f'  loaded {k:30s} {tuple(t.shape)} finite={t.isfinite().all().item()}', flush=True)
    mod = w['action_modality_embed']
    Win = w['action_proj_in.fc.weight'].reshape(NDOM, HID, ACT)
    bin_ = w['action_proj_in.bias.weight']
    Wout = w['action_proj_out.fc.weight'].reshape(NDOM, ACT, HID)
    bout = w['action_proj_out.bias.weight']
    torch.manual_seed(0)
    ok_all = True
    print('\n[action] forward-dynamics (action->token) + inverse-dynamics (token->action) per embodiment:', flush=True)
    for d in (0, 7, 15, 31):
        action_vec = torch.randn(ACT)
        token = Win[d] @ action_vec + bin_[d] + mod
        hidden = torch.randn(HID)
        pred_action = Wout[d] @ hidden + bout[d]
        ok = token.shape == (HID,) and pred_action.shape == (ACT,) and token.isfinite().all().item() and pred_action.isfinite().all().item()
        ok_all = ok_all and ok
        print(f"  domain {d:2d}: action{tuple(action_vec.shape)} -> token{tuple(token.shape)} (|t|={token.norm():.1f})  |  hidden -> action{tuple(pred_action.shape)} (|a|={pred_action.norm():.2f})  [{('OK' if ok else 'FAIL')}]", flush=True)
    print(f"\n=== ACTION MoT HEADS {('FUNCTIONAL' if ok_all else 'FAILED')} ===", flush=True)
    print('  fd/id projection heads (32 embodiment domains, 64-dim actions) load + project finite values.', flush=True)
    print('  Full action-conditioned ROLLOUT (run_fd/run_id) needs the NVIDIA Cosmos framework or vLLM-Omni;', flush=True)
    print('  it is NOT in the diffusers Cosmos3OmniPipeline. The weight-streaming preserves these heads.', flush=True)
    sys.exit(0 if ok_all else 1)
if __name__ == '__main__':
    main()
