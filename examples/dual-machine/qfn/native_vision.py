"""Authenticated loopback image control, reusing DSV's isolated vision worker.

KV/state never enters this Python service. Only image bytes and small projected
image blocks do. The caller holds the GPU lease for this complete lifecycle.
"""
import hmac
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import struct
import subprocess
import threading
import time

from slstack.vision import Images
from slstack.journal import atomic_json,sha256


class NativeVision:
    def __init__(self,cfg,port,token,model,compatibility,retire,reload,lifecycle_lock=None,producer_binary_sha256=None):
        self.cfg=cfg;self.images=Images(cfg);self.retire=retire;self.reload=reload
        self.lock=lifecycle_lock or threading.RLock();self.busy=False;self.closed=False
        if not producer_binary_sha256:raise ValueError('Vision requires the actual paired source executable pin')
        self.producer_binary_sha256=producer_binary_sha256
        self.token=token;self.model=model;self.compatibility=compatibility
        self.events=Path(cfg['runtime_root'])/'vision-events.jsonl'
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*_):pass
            def do_POST(self):
                if self.path!='/vision':self.send_error(404);return
                if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+owner.token):
                    self.send_error(403);return
                if (self.headers.get('X-QFN-Model-SHA256')!=owner.model or
                    self.headers.get('X-QFN-Compatibility-SHA256')!=owner.compatibility or
                    self.headers.get('X-QFN-Producer-Binary-SHA256')!=owner.producer_binary_sha256 or
                    sha256(Path(owner.cfg['qfn_engine']))!=owner.producer_binary_sha256):
                    self.send_error(409);return
                try:
                    size=int(self.headers.get('Content-Length','0'))
                    if not 0<size<=64<<20:raise ValueError('Image body must contain 1..64MiB')
                    raw=self.rfile.read(size)
                    if len(raw)!=size:raise ValueError('Image body ended early')
                    with owner.lock:
                        owner.busy=True;retired=[False];before=[None];start=time.perf_counter()
                        def before_encode():
                            owner.retire();retired[0]=True;before[0]=owner.memory()
                        try:
                            row=owner.images.encode(raw,before_encode)
                            after=owner.memory() if retired[0] else None
                            if retired[0] and after>before[0]+16*1024**2:
                                raise RuntimeError('Vision worker did not return VRAM to its admitted prior level')
                            event=dict(row,image_bytes=size,control_seconds=time.perf_counter()-start,
                                language_retired=retired[0],vram_used_before_worker=before[0],
                                vram_used_after_worker=after,cache_hit=not retired[0])
                            with owner.events.open('a',encoding='utf8') as f:f.write(json.dumps(event)+'\n')
                        finally:
                            try:
                                if retired[0]:owner.reload()
                            finally:owner.busy=False
                        path=Path(row['file']);body_size=path.stat().st_size+16
                        self.send_response(200);self.send_header('Content-Type','application/vnd.studio.qfn-vision-v1')
                        self.send_header('Content-Length',str(body_size));self.end_headers()
                        with path.open('rb') as f:
                            self.wfile.write(f.read(48));self.wfile.write(struct.pack('<IIII',row['grid_height'],row['grid_width'],row['image_height'],row['image_width']))
                            while block:=f.read(1<<20):self.wfile.write(block)
                except Exception as error:
                    raw=json.dumps(dict(error=str(error))).encode()
                    self.send_response(500);self.send_header('Content-Type','application/json')
                    self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        self.server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()

    def memory(self):
        result=subprocess.run(['nvidia-smi','--id='+self.cfg['gpu_uuid'],
            '--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        return int(result.stdout.strip())*1024**2

    def close(self):
        self.closed=True;self.server.shutdown();self.server.server_close();self.thread.join(timeout=5)
