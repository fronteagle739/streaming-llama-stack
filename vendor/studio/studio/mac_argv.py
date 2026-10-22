# Derived from Studio backend/studio/llm_launch.py, ds4_server_argv at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1.
# Exact argv logic; portable fallback binary/directory; original comments/docstring omitted.
import ipaddress
from typing import Optional
DEFAULT_MAC_DS4_SERVER = 'ds4-server'
DEFAULT_MAC_DS4_CHDIR = '.'

def ds4_server_argv(model_path: str, *, port: int, ctx: int, binary: str=DEFAULT_MAC_DS4_SERVER, chdir: str=DEFAULT_MAC_DS4_CHDIR, bind_address: str='127.0.0.1', ssd_streaming: bool=False, ssd_streaming_cache_experts: Optional[str]=None, kv_disk_dir: Optional[str]=None, kv_disk_space_mb: Optional[int]=None, kv_cache_cold_max_tokens: Optional[int]=None, kv_cache_min_tokens: Optional[int]=None, kv_cache_continued_interval_tokens: Optional[int]=None, batched_session: Optional[int]=None, mixed_prefill_quantum: Optional[int]=None, glm_mtp: bool=False, dspark: bool=False, mtp_draft: Optional[int]=None, mtp_timing: bool=False, mtp_model: Optional[str]=None, vision: Optional[str]=None) -> list[str]:
    mtp_model = str(mtp_model).strip() if mtp_model is not None else ''
    vision = str(vision).strip() if vision is not None else ''
    if dspark and (not mtp_model):
        raise ValueError('DS4 ds4_dspark=True requires ds4_mtp_model (the DSpark draft GGUF passed as --mtp-model); refusing to launch --dspark with no draft model')
    if ssd_streaming and mtp_model:
        raise ValueError('DS4 ssd_streaming (--ssd-streaming) is incompatible with ds4_mtp_model (--mtp-model); ds4-server refuses the combination')
    workdir = str(chdir or '').strip()
    if not workdir:
        raise ValueError('DS4 chdir is required so metal/*.metal can be resolved')
    exact_host = str(bind_address or '127.0.0.1').strip()
    try:
        parsed_host = ipaddress.ip_address(exact_host)
    except ValueError as exc:
        raise ValueError('DS4 bind_address must be an exact IPv4 literal') from exc
    if parsed_host.version != 4 or parsed_host.is_unspecified:
        raise ValueError('DS4 bind_address must be an exact non-wildcard IPv4 address')
    argv = [str(binary or DEFAULT_MAC_DS4_SERVER), '--chdir', workdir, '--metal', '-m', str(model_path), '--host', exact_host, '--port', str(int(port)), '--ctx', str(int(ctx))]
    if ssd_streaming:
        argv.append('--ssd-streaming')
    if ssd_streaming_cache_experts is not None:
        budget = str(ssd_streaming_cache_experts).strip()
        if not budget:
            raise ValueError('DS4 SSD streaming expert-cache budget cannot be empty')
        argv += ['--ssd-streaming-cache-experts', budget]
    if kv_disk_space_mb is not None and (not kv_disk_dir):
        raise ValueError('DS4 kv_disk_space_mb is only a budget for --kv-disk-dir; set kv_disk_dir too or the disk KV cache is silently disabled')
    if kv_disk_dir:
        argv += ['--kv-disk-dir', str(kv_disk_dir)]
    if kv_disk_space_mb is not None:
        space = int(kv_disk_space_mb)
        if space <= 0:
            raise ValueError('DS4 KV disk space must be positive')
        argv += ['--kv-disk-space-mb', str(space)]
    if kv_cache_cold_max_tokens is not None:
        cold = int(kv_cache_cold_max_tokens)
        if cold < 0:
            raise ValueError('DS4 KV cold max tokens must be >= 0')
        argv += ['--kv-cache-cold-max-tokens', str(cold)]
    if kv_cache_min_tokens is not None:
        min_tokens = int(kv_cache_min_tokens)
        if min_tokens < 1:
            raise ValueError('DS4 KV cache min tokens must be >= 1')
        argv += ['--kv-cache-min-tokens', str(min_tokens)]
    if kv_cache_continued_interval_tokens is not None:
        interval = int(kv_cache_continued_interval_tokens)
        if interval < 0:
            raise ValueError('DS4 KV cache continued interval must be >= 0 (0 disables)')
        argv += ['--kv-cache-continued-interval-tokens', str(interval)]
    if batched_session is not None:
        sessions = int(batched_session)
        if sessions < 1:
            raise ValueError('DS4 batched-session count must be >= 1')
        argv += ['--batched-session', str(sessions)]
    if mixed_prefill_quantum is not None:
        quantum = int(mixed_prefill_quantum)
        if quantum < 1:
            raise ValueError('DS4 mixed prefill quantum must be >= 1')
        argv += ['--mixed-prefill-quantum', str(quantum)]
    if mtp_timing:
        argv.append('--mtp-timing')
    elif glm_mtp:
        argv.append('--mtp')
    if mtp_model:
        argv += ['--mtp-model', mtp_model]
    if dspark:
        argv.append('--dspark')
    if mtp_draft is not None:
        depth = int(mtp_draft)
        if depth < 1:
            raise ValueError('DS4 MTP draft depth must be >= 1')
        argv += ['--mtp-draft', str(depth)]
    if vision:
        argv += ['--vision', vision]
    return argv
