"""Portable Studio decoder commands and explicit transport capabilities."""
from __future__ import annotations
from dataclasses import dataclass
import json
import ipaddress
import os
from pathlib import Path
import sys
from .config import ROOT, path
from .native_broker import vendor


@dataclass(frozen=True)
class Backend:
    name: str
    transport: str
    setup: str


BACKENDS = {name: Backend(name, transport, setup) for name, transport, setup in (
    ('mlx-receiver', 'kv-token', 'mlx'),
    ('llama-decode', 'kv-token', 'llama'),
    ('llama-decode-upstream', 'kv-token', 'llama-upstream'),
    ('ds4', 'ds4-native', 'ds4'),
    ('ds4-peer', 'ds4-peer', 'ds4-peer'),
    ('mlx-serve', 'openai', 'mlx-serve'),
    ('omlx', 'openai', 'omlx'),
    ('mlx-lm', 'openai', 'mlx'),
    ('mlx-vlm', 'openai', 'mlx'),
    ('mtp-shim', 'openai', 'mlx'),
    ('llama-server', 'openai', 'llama'),
    ('mference', 'openai', 'mference'),
)}
ALIASES = {'gguf-mlx':'mlx-receiver', 'gguf_mlx':'mlx-receiver', 'kimi':'mlx-receiver',
    'kimi-decode':'mlx-receiver', 'decode-receiver':'mlx-receiver',
    'kimi-prompt-direct':'mlx-receiver', 'kimi-direct-shim':'mlx-receiver',
    'remote-kimi':'mlx-receiver', 'kimi-remote':'mlx-receiver',
    'gguf-stream':'llama-decode', 'upstream-llama-decode':'llama-decode-upstream',
    'llama-decode-upstream':'llama-decode-upstream', 'mlx':'mlx-vlm',
    'mlx_serve':'mlx-serve', 'mlxserve':'mlx-serve', 'mlx-core':'mlx-serve',
    'mlxlm':'mlx-lm', 'mlx_lm':'mlx-lm', 'mlxvlm':'mlx-vlm', 'mlx_vlm':'mlx-vlm',
    'glm-mtp-chat':'mtp-shim', 'mference-server':'mference', 'mferenceserver':'mference',
    'swift-moe':'mference', 'dwarfstar':'ds4', 'ds4-server':'ds4', 'llama.cpp':'llama-server',
    'remote-llama':'llama-server', 'llama-upstream':'llama-server'}


def backend(value):
    name=ALIASES.get(str(value).lower(),str(value).lower())
    if name not in BACKENDS:raise ValueError('Unknown decode backend; choose '+', '.join(BACKENDS))
    return BACKENDS[name]


def bind_address(value):
    address=ipaddress.ip_address(str(value))
    if address.is_unspecified or address.is_multicast:raise ValueError('Choose a specific local interface address')
    return str(address)


def python_for(runtime, profile='decode'):
    return str(Path(runtime)/(profile+'-venv')/('Scripts/python.exe' if os.name=='nt' else 'bin/python'))


def installed_binary(runtime,name,fallback):
    receipt=Path(runtime)/'state/backends'/(name+'.json')
    if receipt.is_file():
        from .backend_setup import check
        result=check({'runtime_root':str(runtime)},name)
        if not result.get('ready'):
            raise ValueError('Backend installation is unverified or changed; run setup-backend --backend '+name)
        value=result.get('binary')
        if value:return value
    return str(fallback)


def capabilities(value):
    spec=backend(value)
    return dict(backend=spec.name,transport=spec.transport,rtx_prefill=spec.transport!='openai',
                openai=spec.transport=='openai',setup=spec.setup)


def command(config, *, runtime, python=None, write_settings=True):
    """Build on the serving host. Model paths and binaries stay in local config."""
    vendor()
    from studio import backend_argv as studio
    spec=backend(config.get('backend','mlx-receiver'))
    model=str(config.get('model_path') or '')
    if not model:raise ValueError('Set the decode model_path in your local config')
    host=bind_address(config.get('bind_address','127.0.0.1'))
    port=int(config.get('port',8181));context=int(config.get('ctx',262144))
    if not 1<=port<=65535 or not 1<=context<=1048576:raise ValueError('Invalid decoder port or context')
    py=python or config.get('python') or python_for(runtime,'omlx' if spec.name=='omlx' else 'decode')
    maximum=int(config.get('max_tokens',context))
    sampling={k:config[k] for k in ('temperature','top_p','top_k','min_p') if config.get(k) is not None}
    if spec.name in ('mlx-receiver','mtp-shim'):
        receiver=ROOT/'engines/mlx-receiver/mlx_token_stream_server.py'
        argv=[py,str(receiver),'--model',model,'--host',host,
              '--kv-listen-port',str(int(config.get('kv_port',9100))),
              '--port',str(int(config.get('token_port',9101))),'--max-tokens',str(maximum)]
        for key,flag in (('temperature','--temperature'),('top_p','--top-p'),('top_k','--top-k'),('min_p','--min-p'),
                         ('mtp_mode','--mtp-mode'),('mtp_sidecar','--mtp-sidecar'),('mtp_max_draft','--mtp-max-draft'),
                         ('mtp_token_budget','--mtp-token-budget'),('mtp_adaptive_depth','--mtp-adaptive-depth'),
                         ('mtp_verify_profile','--mtp-verify-profile')):
            if config.get(key) is not None:argv.extend((flag,str(config[key])))
        for key,flag in (('adapters','--adapter'),('token_sidecars','--token-sidecar')):
            for value in config.get(key,[]):argv.extend((flag,str(value)))
        return argv
    if spec.name=='mlx-lm':
        return studio.mlx_lm_argv(model,host=host,port=port,python=py,context=context,max_tokens=maximum,
                                 decode_concurrency=config.get('decode_concurrency',1),prompt_concurrency=config.get('prompt_concurrency',1),**sampling)
    if spec.name=='mlx-vlm':
        return studio.mlx_vlm_argv(model,host=host,port=port,python=py,max_kv_size=context,
            kv_bits=config.get('kv_bits'),draft_model=config.get('draft_model'),draft_kind=config.get('draft_kind'),draft_block_size=config.get('draft_block_size'))
    if spec.name=='mlx-serve':
        from studio.mlx_serve import mlx_serve_argv
        binary=str(config.get('binary') or installed_binary(runtime,spec.name,ROOT/'engines/mlx-serve/zig-out/bin/mlx-serve'))
        argv=mlx_serve_argv({'mode':'remote'},config=config,config_name='mac',port=port,bind_address=host,binary=binary)
        if argv is None:raise FileNotFoundError('Build mlx-serve with slstack mac-setup --backend mlx-serve after staging its pinned native dependencies')
        return argv
    if spec.name=='omlx':
        binary=str(config.get('binary') or Path(py).with_name('omlx'))
        base=str(config.get('base_path') or Path(runtime)/'omlx')
        if not write_settings:
            return [binary,'serve','--model-dir',str(Path(model).parent),'--host',host,'--port',str(port),'--base-path',base]
        return studio.omlx_argv(model,host=host,port=port,python=py,omlx_binary=binary,base_path=base,ctx=context,
            memory_guard=config.get('memory_guard'),max_concurrent_requests=config.get('max_concurrent_requests',1),
            deepseek_v41_engram_ssd_offload=config.get('deepseek_v41_engram_ssd_offload'))
    if spec.name=='mference':
        binary=str(config.get('binary') or installed_binary(runtime,spec.name,ROOT/'engines/mference/.build/release/MferenceServer'))
        return studio.mference_server_argv(model,port=port,binary=binary,bind_address=host,max_context=context,
            model_id=config.get('model_id'),queue_limit=config.get('queue_limit'),prompt_cache_mode=config.get('prompt_cache_mode','single-prefix'),
            expert_cache_slots=config.get('expert_cache_slots'),expert_cache_gb=config.get('expert_cache_gb'),expert_cache_headroom_gb=config.get('expert_cache_headroom_gb'))
    if spec.name=='ds4':
        from studio.mac_argv import ds4_server_argv
        return ds4_server_argv(model,binary=str(config.get('binary') or installed_binary(runtime,spec.name,ROOT/'engines/ds4-mac/ds4-server')),
            chdir=str(ROOT/'engines/ds4-mac'),port=port,ctx=context,dspark=bool(config.get('dspark',False)),
            mtp_model=config.get('mtp_model'),mtp_draft=config.get('mtp_draft'),batched_session=1)
    if spec.name=='ds4-peer':
        from studio.peer_launch import ds4p_peer_argv
        binary=str(config.get('binary') or installed_binary(runtime,spec.name,ROOT/'engines/ds4-peer/ds4-server'))
        mac={**config,'host':host,'server':binary,'http_port':port}
        rtx={'ctx':context,'tokens':maximum,'model_path':model}
        return ds4p_peer_argv(rtx,mac,'unused-rtx-binary')[0]
    if spec.name in ('llama-decode','llama-decode-upstream'):
        name='llama-upstream-decode-receiver' if spec.name.endswith('upstream') else 'llama-decode-receiver'
        binary=str(config.get('binary') or installed_binary(runtime,spec.name,ROOT/'build/mac-llama/bin'/name))
        argv=[binary,'-m',model,'-ngl',str(config.get('ngl',99)),'-c',str(context),'-n',str(maximum),
              '--kv-host',host,'--token-stream-host',host,
              '--kv-port',str(int(config.get('kv_port',9100))),'--token-stream-port',str(int(config.get('token_port',9101)))]
        for key,flag in (('batch','-b'),('ubatch','-ub'),('ctk','-ctk'),('ctv','-ctv')):
            if config.get(key) is not None:argv.extend((flag,str(config[key])))
        if config.get('fa')=='off':argv.append('--no-fa')
        elif config.get('fa')=='on':argv.append('--flash-attn')
        return argv
    binary=str(config.get('binary') or installed_binary(runtime,spec.name,ROOT/'build/mac-llama/bin/llama-server'))
    argv=[binary,'-m',model,'--host',host,'--port',str(port),'-c',str(context),'-ngl',str(config.get('ngl',99)),'--jinja']
    from studio.preset_samplers import launch_args
    argv+=launch_args(config)
    if config.get('jinja_file'):argv.extend(('--chat-template-file',str(config['jinja_file'])))
    for key,flag in (('batch','-b'),('ubatch','-ub'),('ctk','-ctk'),('ctv','-ctv'),('fa','-fa')):
        if config.get(key) is not None:argv.extend((flag,str(config[key])))
    return argv


def commands(config, *, runtime, python=None, write_settings=True):
    primary=command(config,runtime=runtime,python=python,write_settings=write_settings)
    if backend(config.get('backend','mlx-receiver')).name!='mtp-shim':return [primary]
    host=bind_address(config.get('bind_address','127.0.0.1'))
    return [primary,[python or config.get('python') or python_for(runtime),
        str(ROOT/'engines/mlx-receiver/mtp_openai_shim.py'),
        '--receiver-host',host,'--receiver-port',str(config.get('token_port',9101)),
        '--host',host,'--port',str(config.get('port',8181)),
        '--model-id',str(config.get('model_id') or Path(config['model_path']).name),
        '--model-dir',str(config['model_path'])]]
