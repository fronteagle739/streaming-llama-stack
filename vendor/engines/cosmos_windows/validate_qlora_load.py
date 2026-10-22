# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/validate_qlora_load.py
# Machine fallback paths cleared; no model data included.
import sys, time, argparse, threading

def _gpu_used_mib(idx=1):
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(idx)).used // (1024 * 1024)
    except Exception:
        return -1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='')
    ap.add_argument('--rank', type=int, default=32)
    ap.add_argument('--bits', type=int, default=4, choices=[4, 8])
    a = ap.parse_args()
    import torch
    from pathlib import Path
    tdir = str(Path(a.model) / 'transformer')
    try:
        import peft
        import bitsandbytes
    except Exception as e:
        print(f'[qlora] DEP MISSING: {type(e).__name__}: {e}  -> pip install peft bitsandbytes')
        print('RESULT BLOCKED')
        return 2
    from diffusers import BitsAndBytesConfig
    from diffusers.models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer
    base = _gpu_used_mib()
    print(f'[qlora] loading {a.bits}-bit frozen base from {tdir} (GPU base used={base} MiB)...', flush=True)
    if a.bits == 4:
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    else:
        qcfg = BitsAndBytesConfig(load_in_8bit=True)
    t0 = time.time()
    try:
        tf = Cosmos3OmniTransformer.from_pretrained(tdir, quantization_config=qcfg, torch_dtype=torch.bfloat16)
    except Exception as e:
        print(f'[qlora] LOAD FAILED ({a.bits}-bit, SM120/Windows): {type(e).__name__}: {str(e)[:300]}')
        print('RESULT FAIL_LOAD')
        return 1
    load_s = time.time() - t0
    after_load = _gpu_used_mib()
    print(f'[qlora] loaded in {load_s:.1f}s; GPU used={after_load} MiB (+{after_load - base} MiB base)', flush=True)
    tf.requires_grad_(False)
    from peft import LoraConfig
    targets = ['add_q_proj', 'add_k_proj', 'add_v_proj', 'to_add_out']
    try:
        tf.add_adapter(LoraConfig(r=a.rank, lora_alpha=a.rank, init_lora_weights='gaussian', target_modules=targets))
    except Exception as e:
        print(f'[qlora] add_adapter FAILED: {type(e).__name__}: {str(e)[:300]}')
        print('RESULT FAIL_ADAPTER')
        return 1
    from diffusers.training_utils import cast_training_params
    cast_training_params(tf, dtype=torch.float32)
    trainable = sum((p.numel() for p in tf.parameters() if p.requires_grad))
    total = sum((p.numel() for p in tf.parameters()))
    after_adapter = _gpu_used_mib()
    print(f'[qlora] LoRA attached: trainable={trainable / 1000000.0:.1f}M / total={total / 1000000000.0:.2f}B ({100 * trainable / total:.3f}%); GPU used={after_adapter} MiB', flush=True)
    n_adapter = sum((1 for n, _ in tf.named_parameters() if 'lora_' in n))
    print(f'[qlora] adapter tensors: {n_adapter} (expect 2 x {len(targets)} x n_layers)', flush=True)
    print(f'[qlora] gradient_checkpointing supported: {tf._supports_gradient_checkpointing}', flush=True)
    print('RESULT PASS', flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
