"""In-process GLM A/B, callable by the receiver's authorized profiling hook.

Accepts an ALREADY LOADED model. Never imports a model loader or changes presets.
This file does not install/arm the external hook; that is outside F5 ownership.
"""
import importlib
import json
import os
from pathlib import Path
import statistics
import time


def run(model, prompt_ids, output, *, prompt_tokens=4096, steps=32):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    lang=getattr(model,'language_model',model)
    if 'glm' not in (type(model).__module__+' '+type(lang).__module__).lower():
        raise RuntimeError(f'Expected already loaded GLM, got {type(model)} / {type(lang)}')
    if not prompt_ids:
        raise ValueError('Need at least one prompt token')
    modules=[importlib.import_module('mlx_vlm.custom_kernels.glm_moe_dsa.'+n)
             for n in ('fast','kernels','sparse_mla')]
    ops=importlib.import_module('mlx_vlm.models.glm_moe_dsa.native_ops')
    # Restore every module dictionary and the environment, including on failure.
    # This avoids changing the live serving policy after the measurement.
    saved=[(m,dict(m.__dict__)) for m in modules+[ops]]
    old=os.environ.get('MLX_GLM_DSA_NATIVE_KERNELS')
    result=dict(model=str(type(model)),lang=str(type(lang)),runs=[],steps=steps,
                prompt_tokens=prompt_tokens,policy='separate caches; warm/off/on/on/off; no model reload')
    ids=(list(prompt_ids)*((prompt_tokens+len(prompt_ids)-1)//len(prompt_ids)))[:prompt_tokens]
    try:
        os.environ['MLX_GLM_DSA_NATIVE_KERNELS']='1'
        for m in modules+[ops]: importlib.reload(m)
        if not ops.native_available(): raise RuntimeError(f'Native unavailable: {ops.status()}')
        result['native_status']=ops.status()
        counts={}
        for name in ('fused_indexer_scores','dsa_topk_indices'):
            fn=getattr(ops,name)
            def counted(*a,_fn=fn,_name=name,**kw):
                out=_fn(*a,**kw)
                key=_name+('.native' if out is not None else '.unsupported_or_disabled')
                counts[key]=counts.get(key,0)+1
                return out
            setattr(ops,name,counted)
        for enabled in (False,True,False,True,True,False):
            os.environ['MLX_GLM_DSA_NATIVE_KERNELS']='1' if enabled else '0'
            cache=lang.make_cache() if hasattr(lang,'make_cache') else make_prompt_cache(lang)
            t=time.perf_counter()
            for begin in range(0,len(ids),256):
                out=lang(mx.array([ids[begin:begin+256]],mx.int32),cache=cache)
                logits=getattr(out,'logits',out)
                token=mx.argmax(logits[:,-1],axis=-1)
                mx.eval(token,[c.state for c in cache])
            prefill=time.perf_counter()-t
            counts.clear(); generated=[]; t=time.perf_counter()
            for _ in range(steps):
                out=lang(token[:,None],cache=cache)
                logits=getattr(out,'logits',out)
                token=mx.argmax(logits[:,-1],axis=-1)
                mx.eval(token,[c.state for c in cache]); generated.append(token.item())
            elapsed=time.perf_counter()-t
            result['runs'].append(dict(enabled=enabled,prefill_s=prefill,decode_tok_s=steps/elapsed,
                                       calls=dict(counts),tokens=generated))
            if enabled and not any(k.endswith('.native') and v for k,v in counts.items()):
                raise RuntimeError('Extension loaded but no native calls during decode')
            Path(output).write_text(json.dumps(result,indent=2))
        for enabled in (False,True):
            result['native' if enabled else 'baseline']=statistics.median(
                r['decode_tok_s'] for r in result['runs'][2:] if r['enabled']==enabled)
        result['generated_tokens_equal']=all(r['tokens']==result['runs'][0]['tokens'] for r in result['runs'])
        return result
    finally:
        for module,state in saved:
            module.__dict__.clear(); module.__dict__.update(state)
        if old is None: os.environ.pop('MLX_GLM_DSA_NATIVE_KERNELS',None)
        else: os.environ['MLX_GLM_DSA_NATIVE_KERNELS']=old
        Path(output).write_text(json.dumps(result,indent=2))
