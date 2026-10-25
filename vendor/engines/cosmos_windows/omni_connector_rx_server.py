# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/omni_connector_rx_server.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import os
from pathlib import Path
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', os.environ.get('STUDIO_GPU_INDEX', '0'))
from cosmos_windows.omni_connector_receiver import OmniConnectorReceiver, decode_generation_request, generate_from_request

class DiffusersCudaImagePipeline:

    def __init__(self, model_path: str, flow_shift: float=10.0, offload: bool=True):
        import torch
        from diffusers import Cosmos3OmniPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        self.torch = torch
        self.pipe = Cosmos3OmniPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16, enable_safety_checker=False)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=flow_shift)
        if offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to('cuda')

    def __call__(self, *, prompt, negative_prompt=None, num_frames=1, height=256, width=256, num_inference_steps=35, guidance_scale=6.0, seed=None, enable_sound=False, output_type='pil', enable_safety_check=False, flow_shift=None, **kw):
        gen = None
        if seed is not None:
            gen = self.torch.Generator(device='cuda').manual_seed(int(seed))
        return self.pipe(prompt=prompt, negative_prompt=negative_prompt or '', num_frames=num_frames, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, add_resolution_template=False, add_duration_template=False, generator=gen, output_type=output_type)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=9100)
    ap.add_argument('--from-stage', default='rtx_ar')
    ap.add_argument('--to-stage', default='mac_cosmos')
    ap.add_argument('--put-key', default='ar_conditioning')
    ap.add_argument('--out', default='')
    ap.add_argument('--flow-shift', type=float, default=10.0)
    ap.add_argument('--no-offload', action='store_true', help='keep the model resident on the selected device (faster, more VRAM)')
    ap.add_argument('--once', action='store_true', help='exit after one generation (single live test)')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f'[rx-server] loading Cosmos3 CUDA pipeline from {args.model} ...', flush=True)
    pipeline = DiffusersCudaImagePipeline(args.model, flow_shift=args.flow_shift, offload=not args.no_offload)
    receiver = OmniConnectorReceiver(args.host, args.port)
    receiver.start()
    print(f'[rx-server] LISTENING on {args.host}:{receiver.bound_port}, model loaded — RTX will generate CLEAN on request...', flush=True)
    import time
    while True:
        art = receiver.get_artifact(args.from_stage, args.to_stage, args.put_key, timeout_s=3600.0, remove=True)
        req = decode_generation_request(art)
        if req.num_frames > 1:
            from diffusers.utils import export_to_video
            t0 = time.perf_counter()
            res = pipeline(prompt=req.prompt, negative_prompt=req.negative_prompt, num_frames=req.num_frames, height=req.height, width=req.width, num_inference_steps=req.num_inference_steps, guidance_scale=req.guidance_scale, seed=req.seed, output_type='pil')
            frames = list(getattr(res, 'video', res))
            vid_path = out / f'{req.request_id}.mp4'
            export_to_video(frames, str(vid_path), fps=24)
            print(f'[rx-server] GENERATED VIDEO on RTX -> {vid_path} ({len(frames)} frames, prompt={req.prompt!r}, {time.perf_counter() - t0:.1f}s)', flush=True)
        else:
            result = generate_from_request(req, pipeline, out)
            print(f'[rx-server] GENERATED on RTX -> {result.image_path} (prompt={result.request.prompt!r}, {result.seconds:.1f}s)', flush=True)
        if args.once:
            return 0
if __name__ == '__main__':
    raise SystemExit(main())
