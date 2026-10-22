"""Call Studio's existing backend in an isolated owned worker; no engine copies or mock fallback."""
import json
import os
from pathlib import Path
import sys


def run(request):
    cfg, kind, args = request['cfg'], request['kind'], request['params']
    # Parent sets these BEFORE Python/Studio/torch imports.
    root = Path(cfg['runtime_root']).resolve()
    if Path(os.environ['STUDIO_DATA_ROOT']).resolve() != root / 'studio':
        raise RuntimeError('Studio data isolation was not installed before import')
    from dataclasses import replace
    from studio.config import settings
    from studio.adapter import load_backend
    from studio.models import GenerationParams
    isolated = replace(settings, data_root=root / 'studio', output_dir=root / 'outputs',
                       datasets_dir=root / 'datasets', loras_dir=root / 'loras',
                       models_root=Path(cfg['comfyui_models'] or cfg[kind + '_model']).resolve(),
                       cosmos_windows_path=Path(os.environ['STUDIO_COSMOS_WINDOWS_PATH']))
    for directory in (isolated.data_root, isolated.output_dir, isolated.datasets_dir, isolated.loras_dir):
        directory.mkdir(parents=True, exist_ok=True)
    backend, mock, detail = load_backend(isolated)
    if mock:
        raise RuntimeError('A real Studio backend is required: ' + detail)
    try:
        params = GenerationParams(prompt=args['prompt'], negative=args['negative_prompt'],
                                  width=args['width'], height=args['height'], steps=args['steps'],
                                  guidance=args['guidance'], seed=args['seed'],
                                  model=cfg[kind + '_model'], model_arch=cfg[kind + '_arch'] or None,
                                  model_support_files=cfg[kind + '_support_files'],
                                  weight_stream=cfg[kind + '_mode'] == 'streaming',
                                  load_mode='streaming_lazy' if cfg[kind + '_mode'] == 'streaming' else 'resident_cached',
                                  modality='t2i' if kind == 'image' else 't2v',
                                  num_frames=args.get('num_frames', 1), fps=args.get('fps', 24),
                                  output_dir=str(root / 'outputs'))
        result = backend.generate(params)
        data = result.model_dump(mode='json')
        # The pinned Studio GenResult has one primary path (plus optional metadata).
        files = [data['path']] if data.get('path') else []
        if not isinstance(files, list) or not all(isinstance(x, str) for x in files):
            raise RuntimeError('Unsupported Studio result file schema')
        return {'ok': True, 'files': files, 'generation': data}
    finally:
        backend.unload()


if __name__ == '__main__':
    try:
        print('DS4STACK_RESULT ' + json.dumps(run(json.loads(sys.stdin.readline()))), flush=True)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print('DS4STACK_RESULT ' + json.dumps({'ok': False, 'error': str(exc)}), flush=True)
        sys.exit(1)
