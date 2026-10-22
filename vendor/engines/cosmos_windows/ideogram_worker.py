# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/ideogram_worker.py
# Machine fallback paths cleared; no model data included.
import argparse
import json
import sys
import time
_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr

def _emit(obj):
    _REAL_STDOUT.write(json.dumps(obj) + '\n')
    _REAL_STDOUT.flush()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--quant', default='nf4', choices=['nf4', 'fp8'])
    ap.add_argument('--weights-repo', default='')
    a = ap.parse_args()
    import os
    import torch
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    import ideogram_gen as ig
    from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig, PRESETS
    QUANT_REPOS = {'nf4': 'ideogram-ai/ideogram-4-nf4', 'fp8': 'ideogram-ai/ideogram-4-fp8'}
    repo = a.weights_repo or QUANT_REPOS[a.quant]
    _emit({'ready': True, 'quant': a.quant, 'repo': repo})
    pipe = {'p': None}

    def _ensure_pipe():
        if pipe['p'] is not None:
            return pipe['p']
        from pathlib import Path as _P
        if _P(str(repo)).exists():
            try:
                _hub = ig._ideo_pipeline.hf_hub_download

                def _local(*args, **kw):
                    rid = kw.pop('repo_id', args[0] if args else None)
                    fn = kw.pop('filename', args[1] if len(args) > 1 else None)
                    root = _P(str(rid))
                    if root.exists() and fn:
                        cand = root.joinpath(*[p for p in str(fn).replace('\\', '/').split('/') if p])
                        if cand.is_file():
                            return str(cand)
                        raise ig._ideo_pipeline.EntryNotFoundError(f'missing {cand}')
                    return _hub(*args, **kw) if args else _hub(repo_id=rid, filename=fn, **kw)
                ig._ideo_pipeline.hf_hub_download = _local
            except Exception:
                pass
        t0 = time.time()
        p = Ideogram4Pipeline.from_pretrained(config=Ideogram4PipelineConfig(weights_repo=repo), device='cuda', dtype=torch.bfloat16)
        pipe['p'] = p
        _emit({'loaded': True, 'seconds': round(time.time() - t0, 1)})
        return p

    def _gen(req):
        p = _ensure_pipe()
        preset = req.get('preset') or 'V4_QUALITY_48'
        if preset not in PRESETS:
            preset = 'V4_QUALITY_48'
        pr = PRESETS[preset]
        explicit = bool(req.get('steps'))
        num_steps = int(req.get('steps') or pr.num_steps)
        gsched = ig._guidance_schedule_for_steps(preset, pr.guidance_schedule, num_steps, explicit)
        if req.get('image'):
            imgs, _ = ig._ideogram_img2img(p, req['prompt'], init_image_path=str(req['image']), mask_path=str(req['mask']) if req.get('mask') else None, denoise=req.get('denoise'), height=int(req['height']), width=int(req['width']), num_steps=num_steps, guidance_schedule=gsched, mu=pr.mu, std=pr.std, seed=int(req.get('seed', 42)), raise_on_caption_issues=False, release_vae_after_encode=False)
        else:
            imgs = p(req['prompt'], height=int(req['height']), width=int(req['width']), num_steps=num_steps, guidance_schedule=gsched, mu=pr.mu, std=pr.std, seed=int(req.get('seed', 42)), raise_on_caption_issues=False)
        img = imgs[0] if isinstance(imgs, (list, tuple)) else imgs
        img.save(str(req['out']))
        return req['out']
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            _emit({'ok': False, 'error': f'bad json: {e}'})
            continue
        cmd = req.get('cmd')
        if cmd == 'quit':
            _emit({'ok': True, 'bye': True})
            return 0
        if cmd == 'ping':
            _emit({'ok': True, 'pong': True, 'loaded': pipe['p'] is not None})
            continue
        try:
            path = _gen(req)
            _emit({'ok': True, 'path': path})
        except Exception as e:
            import traceback
            _emit({'ok': False, 'error': f'{type(e).__name__}: {e}', 'tb': traceback.format_exc()[-1500:]})
    return 0
if __name__ == '__main__':
    sys.exit(main())
