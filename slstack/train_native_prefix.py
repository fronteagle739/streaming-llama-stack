"""Owned DSV4.1 prefix-provider lifecycle around Studio's streamed trainer.

Run only as a guarded GPU child. All training, dequantization, optimizer and
checkpoint logic stays in the vendored Studio/streaming trainer.
"""
import json
import sys
import time
import socket
from pathlib import Path
from .config import load, path
from .engines import Engines
from .journal import atomic_json, sha256
from .process import OwnedProcess


def run(request):
    cfg, settings = request['cfg'], request['settings']
    cfg['_gpu_lease_managed'] = True
    cfg['lora'] = []
    command = list(cfg['ds4_command'])
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    command[command.index('--port') + 1] = str(port)
    cfg['ds4_url'] = 'http://127.0.0.1:' + str(port)
    cache_flag = '--ssd-streaming-cache-experts'
    if cache_flag not in command:
        raise ValueError('Native prefix training requires the existing SSD streaming cache argument')
    command[command.index(cache_flag) + 1] = '26GB'
    command.append('--training-features')
    if '--ctx-size' in command:
        command[command.index('--ctx-size') + 1] = '256'
    cfg['ds4_command'] = command
    model = path(settings.get('model', cfg['model']))
    cfg['model'] = str(model)
    output = Path(request['output'])
    before_started = time.monotonic()
    before = sha256(model)
    before_s = time.monotonic() - before_started
    if settings.get('expected_base_sha256') and settings['expected_base_sha256'] != before:
        raise RuntimeError('Configured training base SHA-256 does not match the model bytes')
    engine = Engines(cfg)
    trainer = None
    started = time.monotonic()
    try:
        engine.ready()
        token_file = Path(cfg['runtime_root']) / 'state' / ('train-' + engine.llm.prefix.name + '.token')
        token_file.parent.mkdir(parents=True, exist_ok=True)
        with token_file.open('x', encoding='ascii') as stream:
            stream.write(cfg['_control_token'])
        token_file.chmod(0o600)
        from types import SimpleNamespace
        from studio.streamed_lora_argv import streamed_lora_train_argv
        defaults = dict(steps=60, lr=0.002, rank=8, alpha=None, lora_layers=12, batch=40,
                        dataset_size=40, max_seq=64, seed=0, checkpoint_every=0)
        defaults.update({k: v for k, v in settings.items() if k in defaults})
        recipe = SimpleNamespace(**{'quant_stream_' + k: v for k, v in defaults.items()},
            quant_stream_arch='deepseek41', quant_stream_mix_neutral=0,
            quant_stream_dataset=request['dataset'], quant_stream_gpu_buffers=3,
            quant_stream_eval_tokens=8, quant_stream_rss_limit_gb=cfg['host_budget_gib'],
            quant_stream_lr_phases='', quant_stream_prefix_cache=True,
            quant_stream_directstorage=True, quant_stream_suffix_ds=False,
            quant_stream_no_eval=True, quant_stream_selftest=False,
            quant_native_feature_url=cfg['ds4_url'], quant_native_feature_token_file=str(token_file),
            quant_native_base_sha256=before)
        argv = streamed_lora_train_argv(recipe, python=cfg['studio_python'],
            trainer=path(cfg['streaming_root']) / 'cosmos_windows/gguf_qat/train_gguf_lora_streamed.py',
            gguf=str(model), save_dir=output, resume=settings.get('resume', ''))
        trainer = OwnedProcess(cfg, argv, 'studio-dsv41-output-trainer',
                               cwd=path(cfg['streaming_root']), timeout=cfg['lease_minutes'] * 60)
        receipt = trainer.wait()
        if not (output / 'adapter_model.safetensors').is_file():
            raise RuntimeError('Streamed trainer returned without a Safetensors adapter')
    finally:
        if trainer:
            trainer.stop()
        engine.close()
        after_started = time.monotonic()
        after = sha256(model)
        if after != before:
            raise RuntimeError('BASE HASH MISMATCH after native-prefix training')
    result = {'schema': 'slstack.native_prefix_training.v1', 'base_sha256_before': before,
              'base_sha256_after': after, 'base_unchanged': True,
              'target': 'output.weight', 'architecture': 'deepseek41',
              'prefix': 'original frozen quantized native MoE/MLA/HC',
              'training_seconds': after_started - started,
              'hash_before_seconds': before_s, 'hash_after_seconds': time.monotonic() - after_started,
              'trainer_receipt': receipt, 'trainer_log': str(trainer.log),
              'adapter_sha256': sha256(output / 'adapter_model.safetensors')}
    atomic_json(output / 'training_receipt.json', result)
    return result


if __name__ == '__main__':
    request = json.loads(sys.stdin.readline())
    print('SLSTACK_TRAIN_RESULT ' + json.dumps(run(request)), flush=True)
