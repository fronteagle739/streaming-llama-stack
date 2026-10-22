import json
from pathlib import Path
import time
from .engines import http_json
from .journal import atomic_json
from .lora import output_path


def run_prompts(cfg, folder, output, endpoint=None, model=None, max_tokens=256):
    folder = Path(folder).resolve(strict=True)
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ('.txt', '.json'))
    if not files:
        raise ValueError('No .txt or .json prompts in folder')
    out = output_path(cfg, output)
    out.mkdir()
    for index, file in enumerate(files):
        if file.stat().st_size > 4 * 1024**2:
            raise ValueError('Prompt file exceeds 4 MiB limit')
        text = file.read_text(encoding='utf-8-sig')
        data = json.loads(text) if file.suffix.lower() == '.json' else {'messages': [{'role': 'user', 'content': text}]}
        data.update(model=model or cfg['model_name'], stream=False, max_tokens=max_tokens)
        start = time.monotonic()
        result = http_json((endpoint or f'http://{cfg["host"]}:{cfg["port"]}').rstrip('/') + '/v1/chat/completions',
                           data, cfg['request_timeout_s'])
        atomic_json(out / f'{index:04d}-{file.stem}.json', {'prompt_file': str(file),
                    'elapsed_s': time.monotonic() - start, 'response': result})
    return {'output': str(out), 'count': len(files)}
