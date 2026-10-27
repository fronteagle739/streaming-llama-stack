"""WebUI route for general RTX prefill and Studio's Mac HTTP backends."""
from __future__ import annotations
import json
import os
from pathlib import Path
import socket
import sys
import time
from .backends import backend,installed_binary
from .config import ROOT,path
from .engines import Engines,http_json
from .native_broker import vendor
from .process import OwnedProcess
from .progress import Window,activity,readiness_seconds
from .streaming import events


class Remote:
    def __init__(self,url,headers,model):
        self.url=url.rstrip('/');self.headers=headers;self.model=model;self.connected=True
    def body(self,body):return {**body,'model':self.model}
    def chat(self,body):return http_json(self.url+'/v1/chat/completions',self.body(body),None,headers=self.headers)
    def events(self,body):
        yield from events(self.url+'/v1/chat/completions',{**self.body(body),'stream':True},timeout=None,headers=self.headers)
    def close(self):self.connected=False


def launch_spec(cfg):
    vendor()
    if cfg['general'].get('mode')=='peer':
        from studio.peer_launch import ds4p_peer_argv
        general=cfg['general'];rtx=dict(general.get('rtx') or {});mac=general.get('mac') or {}
        if backend(mac.get('backend','ds4-peer')).name!='ds4-peer':raise ValueError('Peer mode requires the ds4-peer decoder')
        rtx['model_path']=str(path(cfg['model']))
        binary=general.get('server') or installed_binary(cfg['runtime_root'],'ds4-peer-rtx',ROOT/'engines/ds4-win/ds4-server.exe')
        argv=ds4p_peer_argv(rtx,mac,str(path(binary)),host=cfg['host'],port=int(general.get('rtx_port',8094)))[1]
        env={'DS4_PEER_IO_TIMEOUT_SEC':str(int(rtx.get('peer_io_timeout_sec',120))),
             'DS4_CUDA_DS_GPU_DEST':'1','DS4_CUDA_ATTN_COMP_CACHE_Q8':'0'}
        if cfg.get('gpu_uuid'):env['CUDA_VISIBLE_DEVICES']=cfg['gpu_uuid']
        return argv,env,f"http://{cfg['host']}:{int(general.get('rtx_port',8094))}"
    from studio.general_launch import streaming_prefill_env,streaming_prefill_argv,stream_arch_id,_resolve_gguf_model_path
    from studio.gguf_meta import _read_kv_header
    from .general_templates import template_file
    general=cfg['general'];rtx=dict(general.get('rtx') or {});mac=general.get('mac') or {}
    spec=backend(mac.get('backend','mlx-receiver'))
    if spec.transport!='kv-token':raise ValueError(spec.name+' uses its own HTTP prefill; choose general.mode=remote')
    if not mac.get('host'):raise ValueError('Set general.mac.host to the Mac receiver address')
    rtx['model_path']=_resolve_gguf_model_path(str(path(cfg['model'])))
    metadata=_read_kv_header(Path(rtx['model_path'])) if Path(rtx['model_path']).is_file() else {}
    architecture=(metadata or {}).get('general.architecture','')
    if architecture:
        source=ROOT/'engines/streaming-llama/prefill_llama.cpp/src/llama-arch.cpp'
        if ('"'+architecture+'"') not in source.read_text(encoding='utf-8'):
            raise ValueError('The retained RTX producer has no graph for '+architecture+'; choose a compatible remote backend')
    if not rtx.get('arch'):rtx['arch']=stream_arch_id(architecture)
    if not rtx.get('arch'):raise ValueError('Set general.rtx.arch for this model architecture')
    if not rtx.get('remote_layers'):
        rtx['remote_layers']=int((metadata or {}).get(architecture+'.block_count',0))-int((metadata or {}).get(architecture+'.nextn_predict_layers',0))
    if rtx['remote_layers']<=0:raise ValueError('Set general.rtx.remote_layers to the target decoder layer count')
    template=rtx.pop('jinja_file',None)
    prepared=template_file(cfg['runtime_root'],inline=rtx.get('jinja'),filename=str(path(template)) if template else None,metadata=metadata)
    if prepared:rtx['jinja']=prepared
    server=str(path(general.get('server') or installed_binary(cfg['runtime_root'],'rtx-producer',ROOT/'build/llama/bin/llama-server.exe')))
    host=cfg['host'];port=int(general.get('rtx_port',8094))
    argv=streaming_prefill_argv(rtx,mac,server,host=host,port=port)
    env=streaming_prefill_env(rtx,mac)
    if cfg.get('gpu_uuid'):env['CUDA_VISIBLE_DEVICES']=cfg['gpu_uuid']
    env['LLAMA_KV_CACHE_DIR']=str(Path(cfg['runtime_root'])/'prefill-cache')
    env.setdefault('LLAMA_KV_CACHE_MAX_GB','16')
    return argv,env,f'http://{host}:{port}'


class GeneralEngines(Engines):
    def single_engine(self):return False
    def combo_enabled(self):return False
    def context_size(self):
        general=self.cfg['general'];mode=general.get('mode','streaming')
        section=general.get('mac' if mode=='remote' else 'rtx') or {}
        return int(section.get('ctx',8192 if mode=='streaming' else 262144))
    def quality(self,value=None):
        if value is not None:raise ValueError('This decoder does not expose the DSV4.1 live affinity control')
        return dict(supported=False,phases_supported=False,reason='Sampling is passed to the selected backend on each request')
    def close(self):
        if self.remote:self.remote.close();self.remote=None
        if self.llm:
            self.llm.stop();self.llm=None
        self.loading_model=None
    def ready(self):
        with self.lock:
            if self.remote and self.remote.connected and (not self.llm or self.llm.proc.poll() is None):return
            if self.llm and self.llm.proc.poll() is not None:self.close()
            general=self.cfg['general'];mac=general.get('mac') or {}
            mode=general.get('mode','streaming')
            headers={}
            if general.get('api_key'):headers['Authorization']='Bearer '+general['api_key']
            if mode=='remote':
                endpoint=str(general.get('endpoint') or '')
                if not endpoint:raise ValueError('Set general.endpoint to the selected Mac HTTP server')
                rows=http_json(endpoint.rstrip('/')+'/v1/models',timeout=10,headers=headers).get('data',[])
                model=mac.get('model_id') or (rows[0]['id'] if rows else self.selected_model)
                self.remote=Remote(endpoint,headers,model)
                self.status('Mac '+backend(mac.get('backend','mlx-serve')).name+' ready')
                return
            if mode not in ('streaming','peer'):raise ValueError('general.mode must be streaming, peer or remote')
            argv,env,url=launch_spec(self.cfg)
            self.status('Loading general RTX streaming prefill')
            self.loading_model=self.selected_model
            # The outer router owns the full tree budget. The inner guardian
            # owns the producer's GPU hold and its original memory/floor limits.
            launch={**self.cfg,'env_overrides':{**self.cfg.get('env_overrides',{}),**env},
                    'host_budget_gib':self.cfg.get('general_producer_host_budget_gib',self.cfg['host_budget_gib']-.75)}
            self.llm=OwnedProcess(launch,argv,'general-prefill',cwd=ROOT,gpu=True,
                                  scope='ds4-win' if mode=='peer' else 'rtx-producer',unbounded=True)
            window=Window(readiness_seconds(self.cfg))
            try:
                while not window.expired():
                    if self.llm.proc.poll() is not None:self.llm.wait();raise RuntimeError('General producer exited before readiness')
                    try:
                        rows=http_json(url+'/v1/models',timeout=2).get('data',[])
                        self.remote=Remote(url,{},rows[0]['id'] if rows else self.selected_model)
                        self.loading_model=None;self.status('RTX prefill and Mac '+backend(mac.get('backend','mlx-receiver')).name+' ready')
                        return
                    except (OSError,RuntimeError):
                        window.observe(activity(self.llm.proc.pid,self.llm.log));time.sleep(.2)
                raise TimeoutError('General producer stopped making startup progress')
            except BaseException:self.close();raise
    def chat(self,body):
        with self.lock:
            try:
                self.select(body.get('model'));self.ready();return self.remote.chat(body)
            finally:
                if self.cfg['general'].get('mode','streaming')!='remote':self.close()
    def chat_events(self,body):
        # Bound the GPU reservation to a request, never the WebUI service's
        # lifetime. Cleanup also runs when a browser disconnects or cancels.
        with self.lock:
            try:
                self.select(body.get('model'));self.ready();yield from self.remote.events(body)
            finally:
                if self.cfg['general'].get('mode','streaming')!='remote':self.close()
