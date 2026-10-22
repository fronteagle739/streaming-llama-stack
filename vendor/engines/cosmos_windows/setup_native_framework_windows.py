# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/setup_native_framework_windows.py
# Machine fallback paths cleared; no model data included.
import os, sys, subprocess, sysconfig, pathlib, json, shutil
FW = sys.argv[1] if len(sys.argv) > 1 else ''
PY = sys.executable
SITE = sysconfig.get_paths()['purelib']
DEPS = ['cattrs', 'hydra-core', 'loguru', 'msgpack', 'obstore', 'omegaconf', 'pydantic', 'scipy', 'termcolor', 'tyro', 'websockets', 'pynvml', 'iopath', 'boto3', 'pandas', 'wandb', 'einops', 'triton-windows', 'qwen-vl-utils']

def pip(*args):
    subprocess.run([PY, '-m', 'pip', 'install', '-q', *args], check=False)

def write_stub(rel, content):
    p = pathlib.Path(SITE) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    print(f'  stub -> {p}')

def main():
    print('[1] install additive deps ...', flush=True)
    pip(*DEPS)
    print('', flush=True)
    subprocess.run([PY, '-m', 'pip', 'uninstall', '-y', '-q', 'multistorageclient'], check=False)
    write_stub('xattr.py', "def getxattr(p,n,*a,**k):\n raise OSError('xattr unsupported on Windows (stub)')\ndef setxattr(p,n,v,*a,**k):\n return None\ndef listxattr(p,*a,**k):\n return []\ndef removexattr(p,n,*a,**k):\n return None\n")
    write_stub('multistorageclient/__init__.py', "class StorageClientConfig:\n    def __init__(self,*a,**k): pass\n    @classmethod\n    def from_dict(cls,*a,**k): return cls()\n    @classmethod\n    def from_file(cls,*a,**k): return cls()\nclass StorageClient:\n    def __init__(self,*a,**k):\n        raise RuntimeError('multistorageclient stubbed on Windows; use local paths')\n")
    write_stub('multistorageclient/types.py', 'class Range:\n    def __init__(self,*a,**k): pass\n')
    print('[3] install diffusers-cosmos3 + transformers-cosmos3 plugins + sample_args ...', flush=True)
    for pkg in ('diffusers-cosmos3', 'transformers-cosmos3'):
        d = os.path.join(FW, 'packages', pkg)
        if os.path.isdir(d):
            pip('--no-deps', '-e', d)
            print(f'  installed {pkg}')
    plugin = os.path.join(FW, 'packages', 'diffusers-cosmos3')
    sa_dst = os.path.join(plugin, 'diffusers_cosmos3', 'sample_args')
    if os.path.isdir(plugin):
        os.makedirs(sa_dst, exist_ok=True)
        for mode in ('text2video', 'text2image', 'image2video', 'image2image', 'video2video', 'forward_dynamics', 'inverse_dynamics', 'policy', 'reasoner'):
            src = os.path.join(FW, 'cosmos_framework', 'inference', 'defaults', mode, 'sample_args.json')
            if os.path.exists(src):
                shutil.copy(src, os.path.join(sa_dst, f'{mode}.json'))
        print(f'  sample_args -> {sa_dst}')
    print('[4] patch download_file symlink->copy (Windows lacks symlink privilege, WinError 1314) ...', flush=True)
    args_py = os.path.join(FW, 'cosmos_framework', 'inference', 'common', 'args.py')
    try:
        txt = pathlib.Path(args_py).read_text(encoding='utf-8')
        if 'path.symlink_to(local_path)\n    meta_path' in txt:
            txt = txt.replace('    path.symlink_to(local_path)\n    meta_path.write_text', '    try:\n        path.symlink_to(local_path)\n    except OSError:\n        import shutil as _sh; _sh.copyfile(local_path, path)\n    meta_path.write_text')
            pathlib.Path(args_py).write_text(txt, encoding='utf-8')
            print('  patched')
        else:
            print('  (already patched or pattern not found — patch manually if needed)')
    except Exception as e:
        print(f'  (patch skipped: {e})')
    print('\n[done] Remaining MANUAL steps (environment-specific):', flush=True)
    print('  * The rope shim cosmos_windows/_cosmos_compat.py must be imported before diffusers_cosmos3.', flush=True)
    print('  * If the HF cache snapshot for nvidia/Cosmos3-Nano is incomplete (only README), junction it to a', flush=True)
    print('    complete local copy (no admin needed):', flush=True)
    print('      Remove-Item <hub>\\models--nvidia--Cosmos3-Nano\\snapshots\\<rev> -Recurse -Force', flush=True)
    print('', flush=True)
    print('  * RUN forward dynamics (direct, no torchrun -> avoids the Windows libuv rendezvous):', flush=True)
    print('      set USE_LIBUV=0 & set RANK=0 & set WORLD_SIZE=1 & set LOCAL_RANK=0 & set MASTER_ADDR=127.0.0.1 &', flush=True)
    print('      set MASTER_PORT=29596 & set CUDA_DEVICE_ORDER=PCI_BUS_ID & set CUDA_VISIBLE_DEVICES=1', flush=True)
    print('      python -m cosmos_framework.scripts.inference --parallelism-preset=throughput --no-guardrails ^', flush=True)
    print('', flush=True)
if __name__ == '__main__':
    main()
