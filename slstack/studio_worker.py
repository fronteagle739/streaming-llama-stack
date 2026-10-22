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
        from studio.custom_models import checkpoint_metadata, comfy_support_defaults, _first_file_any
        defaults = comfy_support_defaults()
        model = cfg[kind + '_model']
        arch = cfg[kind + '_arch']
        if not model and cfg['comfyui_models']:
            folder = Path(cfg['comfyui_models'])
            if kind == 'image':
                arch = 'anima-1.0'
                model = defaults.get(arch, {}).get('base_model')
            else:
                arch = 'ltx-2.3'
                model = defaults.get(arch, {}).get('base_model')
                if not model:
                    arch = 'bernini'
                    model = _first_file_any(folder, ['diffusion_models', 'checkpoints'], ['wan22_bernini_high_mxfp8', 'bernini_high'])
            if not model:
                raise ValueError('No supported default ' + kind + ' checkpoint found in the ComfyUI folder; select an explicit model/architecture')
        if not arch and Path(model).suffix.lower() == '.safetensors':
            arch = checkpoint_metadata(model).get('arch_hint')
        support = {**defaults.get(arch, {}), **cfg[kind + '_support_files']}
        mode = cfg[kind + '_mode']
        if arch and arch.startswith('wan') and mode == 'resident':
            raise ValueError('Pinned Studio Wan adapter exposes its streamed path only; resident mode is unsupported')
        if arch and arch.startswith('bernini') and ((Path(model).is_dir()) != (mode == 'resident')):
            raise ValueError('Pinned Bernini uses resident mode for a diffusers folder and streaming for a single-file checkpoint')
        params = GenerationParams(prompt=args['prompt'], negative=args['negative_prompt'],
                                  width=args['width'], height=args['height'], steps=args['steps'],
                                  guidance=args['guidance'], seed=args['seed'],
                                  model=model, model_arch=arch or None,
                                  model_support_files=support,
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
        return {'ok': True, 'files': files, 'generation': data, 'selected_model': model, 'selected_arch': arch}
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
