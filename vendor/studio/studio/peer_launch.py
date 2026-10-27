from __future__ import annotations
import sys
from .general_launch import _resolve_gguf_model_path

def ds4p_peer_argv(rtx: dict, mac: dict, server: str, *, host: str='127.0.0.1', port: int=8080) -> tuple[list, list]:
    mac_host = mac.get('host') or ''
    if not mac_host:raise ValueError('Set the Mac peer interface address')
    peer_port = int(mac.get('peer_port', 49173))
    mac_http = int(mac.get('http_port', 18961))
    ctx = int(rtx.get('ctx', 262144))
    tokens = int(rtx.get('tokens', 2048))
    mac_model = str(mac.get('model_path') or '')
    rtx_model = _resolve_gguf_model_path(rtx.get('model_path', ''))
    if not mac_model:
        raise RuntimeError("ds4p_peer_argv: mac.model_path is required (the decode side's gguf)")
    mac_argv = [mac.get('server', './ds4-server'), '--metal', '-m', mac_model, '--host', '127.0.0.1', '--port', str(mac_http), '--ctx', str(ctx), '--tokens', str(tokens), '--peer-listen', '%s:%d' % (mac_host, peer_port), '--peer-session-ttl', str(int(mac.get('peer_session_ttl', 900))), '--peer-session-max', str(int(mac.get('peer_session_max', 4)))]
    mac_argv += _ds4_kv_disk_args(mac.get('kv_disk_dir') or mac.get('ds4_kv_disk_dir'), mac.get('kv_disk_space_mb') or mac.get('ds4_kv_disk_space_mb'), side='mac')
    rtx_argv = [server, '--cuda', '-m', rtx_model, '--host', host, '--port', str(port), '--ctx', str(ctx), '--ssd-streaming', '--ssd-streaming-cache-experts', str(rtx.get('ssd_cache_experts', '28GB')), '--ram-streaming-cache-experts', str(rtx.get('ram_cache_experts', '12GB')), '--peer', '%s:%d' % (mac_host, peer_port), '--peer-role', 'decode']
    rtx_argv += _ds4_kv_disk_args(rtx.get('ds4_kv_disk_dir') or rtx.get('kv_disk_dir'), rtx.get('ds4_kv_disk_space_mb') or rtx.get('kv_disk_space_mb'), side='rtx')
    return (mac_argv, rtx_argv)

def _ds4_kv_disk_args(kv_dir, space_mb, *, side: str) -> list:
    d = str(kv_dir or '').strip()
    if not d:
        if space_mb:
            print(f'WARNING: DS4P {side}: kv_disk_space_mb={space_mb} without a kv_disk_dir enables nothing; ignored', file=sys.stderr, flush=True)
        return []
    out = ['--kv-disk-dir', d]
    if space_mb:
        mb = int(space_mb)
        if mb <= 0:
            raise ValueError('DS4P KV disk space must be positive')
        out += ['--kv-disk-space-mb', str(mb)]
    return out
