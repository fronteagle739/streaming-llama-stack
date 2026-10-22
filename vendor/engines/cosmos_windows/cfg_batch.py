# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/cfg_batch.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations

class BlockDiagCosmos3AttnProcessor:

    def __init__(self, orig, und_split: int, gen_split: int):
        self.orig = orig
        self.und_split = und_split
        self.gen_split = gen_split

    def __call__(self, attn, und_seq, gen_seq, rotary_emb):
        import torch
        cu, gu = (self.und_split, self.gen_split)
        cos_u, sin_u, cos_g, sin_g = rotary_emb
        nu, ng = (und_seq.shape[0], gen_seq.shape[0])
        spans = [(0, cu, 0, gu), (cu, nu, gu, ng)]
        u_outs, g_outs = ([], [])
        for us, ue, gs, ge in spans:
            rot_s = (cos_u[us:ue], sin_u[us:ue], cos_g[gs:ge], sin_g[gs:ge])
            u_out, g_out = self.orig(attn, und_seq[us:ue], gen_seq[gs:ge], rot_s)
            u_outs.append(u_out)
            g_outs.append(g_out)
        return (torch.cat(u_outs, dim=0), torch.cat(g_outs, dim=0))

def merge_packs(cond: dict, uncond: dict):
    import torch
    cu, uu = (int(cond['und_len']), int(uncond['und_len']))
    cg, ug = (int(cond['vision_sequence_indexes'].shape[0]), int(uncond['vision_sequence_indexes'].shape[0]))
    und_m = cu + uu
    dev = cond['text_indexes'].device
    input_ids = torch.cat([cond['input_ids'], uncond['input_ids']], dim=cond['input_ids'].ndim - 1)
    text_indexes = torch.arange(und_m, dtype=torch.long, device=dev)
    vision_sequence_indexes = torch.arange(und_m, und_m + cg + ug, dtype=torch.long, device=dev)
    cond_mse = cond['vision_mse_loss_indexes'] + uu
    uncond_mse = uncond['vision_mse_loss_indexes'] + (cu + cg)
    vision_mse_loss_indexes = torch.cat([cond_mse, uncond_mse])
    cp, up = (cond['position_ids'], uncond['position_ids'])
    position_ids = torch.cat([cp[:, :cu], up[:, :uu], cp[:, cu:], up[:, uu:]], dim=1)
    merged = dict(cond)
    merged.update(input_ids=input_ids, text_indexes=text_indexes, und_len=und_m, sequence_length=und_m + cg + ug, position_ids=position_ids, vision_sequence_indexes=vision_sequence_indexes, vision_mse_loss_indexes=vision_mse_loss_indexes, vision_token_shapes=list(cond['vision_token_shapes']) + list(uncond['vision_token_shapes']), vision_noisy_frame_indexes=list(cond['vision_noisy_frame_indexes']) + list(uncond['vision_noisy_frame_indexes']))
    return (merged, cu, cg)

def batched_predict(transformer, cond_kw: dict, uncond_kw: dict):
    import torch
    merged, und_split, gen_split = merge_packs(cond_kw, uncond_kw)
    merged['vision_tokens'] = list(cond_kw['vision_tokens']) + list(uncond_kw['vision_tokens'])
    merged['vision_timesteps'] = torch.cat([cond_kw['vision_timesteps'], uncond_kw['vision_timesteps']])
    if cond_kw.get('sound_tokens') is not None:
        merged['sound_tokens'] = list(cond_kw['sound_tokens']) + list(uncond_kw['sound_tokens'])
        merged['sound_timesteps'] = torch.cat([cond_kw['sound_timesteps'], uncond_kw['sound_timesteps']])
    saved = []
    for layer in transformer.layers:
        attn = layer.self_attn
        saved.append((attn, attn.processor))
        attn.set_processor(BlockDiagCosmos3AttnProcessor(attn.processor, und_split, gen_split))
    try:
        preds_vision, preds_sound = transformer(**merged)
    finally:
        for attn, proc in saved:
            attn.set_processor(proc)
    cv = [preds_vision[0]]
    uv = [preds_vision[1]]
    if preds_sound is not None:
        return (cv, uv, [preds_sound[0]], [preds_sound[1]])
    return (cv, uv, None, None)
_STEP_FIELDS = ('vision_tokens', 'vision_timesteps', 'sound_tokens', 'sound_timesteps')

def enable_cfg_batch(pipe):
    tf = pipe.transformer
    if getattr(tf, '_cfg_batch_on', False):
        reset_cfg_batch(pipe)
        return
    orig_forward = tf.forward
    st = {'n': 0, 'uncond_static': None, 'pending': None}

    def wrapped(**kw):
        if getattr(tf, '_cfg_passthrough', False):
            return orig_forward(**kw)
        is_cond = st['n'] % 2 == 0
        st['n'] += 1
        if is_cond:
            if st['uncond_static'] is None:
                return orig_forward(**kw)
            uncond_kw = {**st['uncond_static'], **{f: kw.get(f) for f in _STEP_FIELDS}}
            tf._cfg_passthrough = True
            try:
                cv, uv, cs, us = batched_predict(tf, kw, uncond_kw)
            finally:
                tf._cfg_passthrough = False
            st['pending'] = (uv, us)
            return (cv, cs)
        else:
            if st['pending'] is not None:
                uv, us = st['pending']
                st['pending'] = None
                return (uv, us)
            out = orig_forward(**kw)
            st['uncond_static'] = {k: v for k, v in kw.items() if k not in _STEP_FIELDS}
            return out
    tf.forward = wrapped
    tf._cfg_batch_on = True
    tf._cfg_state = st
    tf._cfg_orig_forward = orig_forward

def reset_cfg_batch(pipe):
    st = getattr(pipe.transformer, '_cfg_state', None)
    if st is not None:
        st['n'] = 0
        st['uncond_static'] = None
        st['pending'] = None

def disable_cfg_batch(pipe):
    tf = pipe.transformer
    if getattr(tf, '_cfg_batch_on', False):
        tf.forward = tf._cfg_orig_forward
        tf._cfg_batch_on = False

def _selftest_merge():
    import torch
    cu, uu, cg, ug = (5, 3, 7, 7)

    def pack(und, vis, base):
        return {'input_ids': torch.arange(und) + base, 'text_indexes': torch.arange(und), 'und_len': und, 'num_vision_tokens': vis, 'vision_sequence_indexes': torch.arange(und, und + vis), 'vision_mse_loss_indexes': torch.arange(und, und + vis), 'vision_token_shapes': [(1, 1, vis)], 'vision_noisy_frame_indexes': [torch.arange(1)], 'position_ids': torch.arange(und + vis)[None].expand(3, -1).contiguous()}
    cond, uncond = (pack(cu, cg, 0), pack(uu, ug, 100))
    m, us, gs = merge_packs(cond, uncond)
    und_m = cu + uu
    ok = us == cu and gs == cg and (m['und_len'] == und_m) and (m['sequence_length'] == und_m + cg + ug) and torch.equal(m['text_indexes'], torch.arange(und_m)) and torch.equal(m['vision_sequence_indexes'], torch.arange(und_m, und_m + cg + ug)) and torch.equal(m['vision_mse_loss_indexes'][:cg], torch.arange(und_m, und_m + cg)) and torch.equal(m['vision_mse_loss_indexes'][cg:], torch.arange(und_m + cg, und_m + cg + ug)) and (m['position_ids'].shape[1] == und_m + cg + ug) and (len(m['vision_token_shapes']) == 2)
    print(f"merge_packs structural: {('PASS' if ok else 'FAIL')} (und_len={m['und_len']} seq={m['sequence_length']})")
    return ok

def _selftest(device='cpu'):
    import torch
    from diffusers.models.transformers.transformer_cosmos3 import Cosmos3PackedMoTAttention, Cosmos3VLTextRotaryEmbedding
    torch.manual_seed(0)
    H, KV, D, hidden = (4, 2, 8, 32)
    attn = Cosmos3PackedMoTAttention(hidden_size=hidden, head_dim=D, num_attention_heads=H, num_key_value_heads=KV, attention_bias=False, rms_norm_eps=1e-06).to(device).eval()
    orig = attn.processor
    rope = Cosmos3VLTextRotaryEmbedding(head_dim=D, rope_theta=5000000.0, rope_axes_dim=(2, 1, 1)).to(device)

    def rot(n):
        pos = torch.arange(n, device=device)[None, None].expand(3, 1, -1)
        cos, sin = rope(pos, device, torch.float32)
        return (cos.squeeze(0), sin.squeeze(0))
    cu_un, uu_un, gen = (5, 3, 7)
    und_c = torch.randn(cu_un, hidden, device=device)
    gen_c = torch.randn(gen, hidden, device=device)
    und_u = torch.randn(uu_un, hidden, device=device)
    gen_u = torch.randn(gen, hidden, device=device)
    cos_uc, sin_uc = rot(cu_un)
    cos_gc, sin_gc = rot(gen)
    cos_uu, sin_uu = rot(uu_un)
    cos_gu, sin_gu = rot(gen)
    with torch.no_grad():
        ref_uc, ref_gc = orig(attn, und_c, gen_c, (cos_uc, sin_uc, cos_gc, sin_gc))
        ref_uu, ref_gu = orig(attn, und_u, gen_u, (cos_uu, sin_uu, cos_gu, sin_gu))
        bd = BlockDiagCosmos3AttnProcessor(orig, und_split=cu_un, gen_split=gen)
        und_m = torch.cat([und_c, und_u], 0)
        gen_m = torch.cat([gen_c, gen_u], 0)
        cos_um = torch.cat([cos_uc, cos_uu], 0)
        sin_um = torch.cat([sin_uc, sin_uu], 0)
        cos_gm = torch.cat([cos_gc, cos_gu], 0)
        sin_gm = torch.cat([sin_gc, sin_gu], 0)
        bd_u, bd_g = bd(attn, und_m, gen_m, (cos_um, sin_um, cos_gm, sin_gm))
    ref_u = torch.cat([ref_uc, ref_uu], 0)
    ref_g = torch.cat([ref_gc, ref_gu], 0)
    du = (bd_u - ref_u).abs().max().item()
    dg = (bd_g - ref_g).abs().max().item()
    ok = du < 1e-05 and dg < 1e-05
    print(f"block-diag vs 2x separate: max|d_und|={du:.2e} max|d_gen|={dg:.2e} -> {('PASS' if ok else 'FAIL')}")
    return ok

def _selftest_wrapper():

    class FakeTF:

        def __init__(self):
            self.calls = 0

        def forward(self, **kw):
            self.calls += 1
            return ([('orig', kw['und_len'])], None)

    class FakePipe:

        def __init__(self):
            self.transformer = FakeTF()
    pipe = FakePipe()
    tf = pipe.transformer
    bp = {'n': 0, 'uncond_und': None}
    g = globals()
    real_bp = g['batched_predict']

    def fake_bp(transformer, cond_kw, uncond_kw):
        bp['n'] += 1
        bp['uncond_und'] = uncond_kw['und_len']
        return ([('bcond', cond_kw['und_len'])], [('bunc', uncond_kw['und_len'])], None, None)
    g['batched_predict'] = fake_bp
    try:
        enable_cfg_batch(pipe)
        res = []
        for step in range(3):
            res.append(tf.forward(und_len=36, vision_tokens=[step], vision_timesteps=step))
            res.append(tf.forward(und_len=29, vision_tokens=[step], vision_timesteps=step))
    finally:
        g['batched_predict'] = real_bp
    ok = True
    ok &= tf.calls == 2
    ok &= bp['n'] == 2
    ok &= res[0] == ([('orig', 36)], None) and res[1] == ([('orig', 29)], None)
    ok &= res[2] == ([('bcond', 36)], None)
    ok &= res[3] == ([('bunc', 29)], None)
    ok &= res[4] == ([('bcond', 36)], None) and res[5] == ([('bunc', 29)], None)
    reset_cfg_batch(pipe)
    st = tf._cfg_state
    ok &= st['n'] == 0 and st['uncond_static'] is None and (st['pending'] is None)
    print(f"cfg_batch wrapper pairing: tf.calls={tf.calls} batched={bp['n']} unc_und={bp['uncond_und']} -> {('PASS' if ok else 'FAIL')}")
    return ok
if __name__ == '__main__':
    import sys
    ok = _selftest_merge() and _selftest_wrapper() and _selftest()
    sys.exit(0 if ok else 1)
