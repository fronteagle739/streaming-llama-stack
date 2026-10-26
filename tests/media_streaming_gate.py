"""Real media-venv DirectStorage fetch through the GPU ledger guardian."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def child(args):
    import os
    os.environ['STUDIO_WF_DLL']=args.library
    import torch
    from safetensors.torch import save_file
    from cosmos_windows.weight_streaming.safetensors_index import build_index
    from cosmos_windows.weight_streaming.ds_backend import DirectStorageStreamer
    path=Path(args.fixture);expected=torch.arange(65536,dtype=torch.float32).reshape(256,256)
    save_file({'layers.0.weight':expected},str(path))
    stream=DirectStorageStreamer(build_index(path),component='',n_gpu_buffers=1,device='cuda:0',dll_path=args.library)
    try:
        assert stream.mode==3,'DirectStorage GPU destination was not selected'
        actual=stream.wait_layer(0)['layers.0.weight']
        assert actual.is_cuda and torch.equal(actual.cpu(),expected)
        stream.release_layer(0)
        print('SLSTACK_MEDIA_GATE '+json.dumps(dict(status='PASS',mode=stream.mode,values=expected.numel(),all_values_equal=True)),flush=True)
    finally:stream.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config');parser.add_argument('--out');parser.add_argument('--child',action='store_true');parser.add_argument('--library');parser.add_argument('--fixture')
    args=parser.parse_args()
    if args.child:return child(args)
    from slstack.config import load
    from slstack.process import OwnedProcess
    cfg=load(args.config);root=Path(args.out).parent/'qualification-media';root.mkdir(exist_ok=True)
    cfg.update(runtime_root=str(root),host_budget_gib=3,min_available_gib=4,task='MEDIAGATE')
    library=str(Path(cfg['llama_producer']).parent/'slstack-weight-fetcher.dll')
    fixture=root/'streaming-values.safetensors'
    job=OwnedProcess(cfg,[cfg['studio_python'],'-B',__file__,'--child','--library',library,'--fixture',str(fixture)],'image',gpu=True)
    try:
        receipt=job.wait();rows=[json.loads(s.removeprefix('SLSTACK_MEDIA_GATE ')) for s in job.log.read_text(errors='replace').splitlines() if s.startswith('SLSTACK_MEDIA_GATE ')]
        assert len(rows)==1
        proof=dict(rows[0],receipt=str(job.receipt),exit_code=receipt['exit_code'],reason=receipt.get('reason'))
        Path(args.out).write_text(json.dumps(proof,indent=2),encoding='utf-8');print(json.dumps(proof),flush=True)
    finally:
        if job.proc.poll() is None:job.stop()


if __name__=='__main__':main()
