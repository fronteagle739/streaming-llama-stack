"""Guarded, opt-in Q4/MXFP4 comparison using an already verified model receipt.

Supply an experiment config, exact little-endian uint32 prompt tokens, and an
optional fixed continuation. Each invocation takes the shared benchmark lease.
The proxy is lossy; this example never changes a production preset.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slstack.config import load
from slstack.process import OwnedProcess

LABEL = 'Q4\u2192MXFP4 requant (lossy), speed proxy for native FP4'

def graph_witnesses(trace,minimum=1):
    rows=re.findall(r'OTHER22_CUDA_GRAPH launch=1 captured=(\d+) captures=(\d+) replays=(\d+) ledger_entries=(\d+) bounded_executables=1',trace)
    if len(rows)<minimum:
        raise ValueError('Explicit CUDA graph request has insufficient actual launches: '+str(len(rows))+' < '+str(minimum))
    return dict(cuda_graph_launches=len(rows),cuda_graph_captures=int(rows[-1][1]),cuda_graph_replays=int(rows[-1][2]))


def layer_witnesses(trace,tiles,queued=False,fusion=0):
    record={}
    if tiles<=0:raise ValueError('Layer experiments require positive native FFN coverage witnesses')
    if queued:
        certified=re.findall(r'OTHER22_ASYNC_TILE_CERTIFIED layer=(\d+) start=(\d+) count=(\d+)',trace)
        layers=re.findall(r'OTHER22_ASYNC_LAYER_CERTIFIED layer=(\d+) ',trace)
        if len(certified)!=tiles or set(map(int,layers))!=set(range(40)):
            raise ValueError('Queued layer request lacks complete tile/layer certification')
        record.update(asynchronous_certified_tiles=len(certified),asynchronous_certified_layers=len(layers))
    if fusion:
        calls=re.findall(r'OTHER22_LAYER_FUSION rows=(\d+) width=(\d+) HC_BF16_RMS_weight_BF16=1',trace)
        if len(calls)<tiles*2:
            raise ValueError('Layer fusion request lacks both sublayer norm launches for every tile')
        record['layer_fusion_launches']=len(calls)
        if fusion==2:
            graphs=re.findall(r'OTHER22_BINDING_GRAPH launches=(\d+) captures=(\d+) pointer_table=1 mutable_rows=1',trace)
            if len(graphs)<tiles*2 or len(graphs)>len(calls) or int(graphs[-1][0])!=len(graphs) or int(graphs[-1][1])!=1:
                raise ValueError('Pointer-table request did not replay one binding-independent norm graph')
            record.update(binding_graph_launches=len(graphs),binding_graph_captures=int(graphs[-1][1]),
                scoped_eager_fusions=len(calls)-len(graphs))
    return record


def token_ids(path):
    raw = path.read_bytes()
    if not raw or len(raw) % 4:
        raise ValueError('Token file must contain nonempty little-endian uint32 IDs')
    return raw, list(struct.unpack('<' + 'I' * (len(raw) // 4), raw))


def completed_logits_sha256(path):
    # The native forced-decode reply can precede its exclusive Windows file
    # handle's RAII close. Retry only the completed artifact read, never decode.
    deadline = time.monotonic() + 1.0
    while True:
        try:
            with path.open('rb') as stream:
                return hashlib.file_digest(stream, 'sha256').hexdigest()
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.01)


def run(args, cfg):
    from slstack.combo_producer import ComboProducer
    from slstack.native_broker import vendor
    from slstack.gguf_header import read_header
    vendor()
    from studio.dsv41_session_process import Producer

    trace = args.output.with_suffix('.trace.log')

    def drain(process):
        try:
            with trace.open('ab', buffering=0) as stream:
                while block := process.errors.read(65536):
                    stream.write(block)
                    with process.state_lock:
                        process.diagnostic.extend(block)
                        if len(process.diagnostic) > 1 << 20:
                            del process.diagnostic[:len(process.diagnostic) - (1 << 20)]
                    process.last_progress = time.monotonic()
        except BaseException as error:
            process._poison(error)

    Producer._drain = drain
    raw, tokens = token_ids(args.tokens)
    record = dict(status='RUNNING', arm=args.arm,
                  label=LABEL if args.arm == 'proxy' else 'Q4_K/Q8_0 control',
                  prompt_tokens=len(tokens), input_sha256=hashlib.sha256(raw).hexdigest(),
                  profile=cfg['producer_profile'], yoco_keep=cfg['yoco_keep'], bias=args.bias,
                  producer_sha256=cfg['producer_sha256'],
                  decode_library_sha256=hashlib.file_digest(
                      (Path(cfg['llama_producer']).parent / 'ds4-decode.dll').open('rb'), 'sha256').hexdigest(),
                  profiled=args.profile_stages,
                  cuda_graphs_requested=args.cuda_graphs,
                  asynchronous_layers_requested=cfg['env_overrides'].get('LLAMA_DSV41_ASYNC_LAYER')=='1',
                  layer_fusion_requested=int(cfg['env_overrides'].get('LLAMA_DSV41_LAYER_FUSION','0')),
                  native_graph_disable_policy=cfg['env_overrides'].get('GGML_CUDA_DISABLE_GRAPHS'),
                  prefill_only=args.prefill_only,
                  native_ffn_requested=bool(cfg['env_overrides'].get('LLAMA_DSV41_FLASHINFER_NATIVE_FFN_DLL')),
                  dense_fp8_requested=cfg['env_overrides'].get('LLAMA_DSV41_DENSE_FP8')=='1',
                  flashinfer_requested=cfg['env_overrides'].get('LLAMA_DSV41_FLASHINFER_MXFP4') == '1',
                  flashinfer_batch_tokens=int(cfg['env_overrides'].get('LLAMA_DSV41_FLASHINFER_BATCH_TOKENS','4096')),
                  configured_context=cfg['context'],
                  decode_mma_requested=cfg['env_overrides'].get('DS4_CUDA_MXFP4_DECODE_MMA') == '1',
                  route_histogram_requested=cfg['env_overrides'].get('LLAMA_DSV41_ROUTE_HISTOGRAM') == '1',
                  affinity_warmup_tokens=int(cfg['env_overrides'].get('DS4_CACHE_AFFINITY_WARMUP_TOKENS','100')),
                  affinity_ramp_tokens=int(cfg['env_overrides'].get('DS4_CACHE_AFFINITY_RAMP_TOKENS','8')),
                  memory_limits={key: cfg[key] for key in (
                      'host_budget_gib', 'producer_host_budget_gib', 'min_available_gib', 'runtime_min_free_gib')})
    if args.bias:
        record['affinity_calibration_sha256'] = hashlib.file_digest(
            Path(cfg['cache_affinity_spread_file']).open('rb'), 'sha256').hexdigest()
    producer = None
    try:
        if record['dense_fp8_requested']:
            dense_bridge=Path(cfg['env_overrides']['LLAMA_DSV41_DENSE_FP8_DLL'])
            record['dense_fp8_bridge_sha256']=hashlib.file_digest(dense_bridge.open('rb'),'sha256').hexdigest()
            record['label']+=' + Q8_0->FP8 dense (lossy), speed proxy for native FP8'
        if record['flashinfer_requested']:
            if args.arm != 'proxy':
                raise ValueError('FlashInfer stored-weight experiment requires the proxy arm')
            bridge = Path(cfg['env_overrides']['LLAMA_DSV41_FLASHINFER_NATIVE_FFN_DLL'] if record['native_ffn_requested'] else cfg['env_overrides']['LLAMA_DSV41_FLASHINFER_MXFP4_DLL'])
            record['flashinfer_bridge_sha256'] = hashlib.file_digest(bridge.open('rb'), 'sha256').hexdigest()
        header = read_header(cfg['model'])
        record['native_dense_layout']=header['metadata'].get('deepseek41.dense_layout')
        if args.arm=='proxy' and 'deepseek41.speed_proxy' not in header['metadata']:
            record['label']='native E2M1/E8M0 routed experts; reference dense recipe'
            if record['dense_fp8_requested']:record['label']+=' + selected native E4M3/E8M0 dense projections' if record['native_dense_layout'] else ' + Q8_0->FP8 dense (lossy), speed proxy for native FP8'
        if record['native_dense_layout']:
            record['label']=record['label'].replace('Q8_0->FP8 dense (lossy), speed proxy for native FP8','selected native E4M3/E8M0 dense projections')
        experts = [t for t in header['tensors'] if t['name'].endswith(
            ('_gate_exps.weight', '_up_exps.weight', '_down_exps.weight'))]
        if args.arm == 'q4' and (len(experts) != 120 or any(t['type'] != 12 for t in experts)):
            raise ValueError('Q4 control requires exactly 120 original Q4_K expert tensors')
        if args.arm == 'proxy':
            marker=header['metadata'].get('deepseek41.speed_proxy')
            if marker is not None and marker != LABEL:
                raise ValueError('Proxy model marker is absent or incorrect')
            if marker is None and header['metadata'].get('deepseek41.expert_layout')!='flashinfer-sm120-mxfp8-mxfp4-v1':
                raise ValueError('Native source experiment requires the explicit packed expert layout')
            if sum(t['type'] == 39 for t in header['tensors']) != 120:
                raise ValueError('Expected exactly 120 stored MXFP4 expert tensors')
        producer = ComboProducer(cfg, secrets.token_hex(16), 1, tokens,
                                 lambda: args.output.with_suffix('.stop').exists())
        record.update(prefill_seconds=producer.prefill_seconds,
                      prefill_tok_s=len(tokens) / producer.prefill_seconds,
                      plan=producer.prefill_plan, model_sha256=producer.model_sha256,
                      ready=producer.process.ready)
        from timing_report import from_receipt
        record.update(from_receipt(producer.directory / 'producer.json', record))
        if not args.prefill_only:record['quality_reply'] = producer.quality(args.bias)
        if args.continuation:
            _, continuation = token_ids(args.continuation)
            logits = args.output.with_name(args.output.stem + '-' + secrets.token_hex(6) + '-logits.f32')
            record.update(forced_tokens=continuation,
                          forced_reply=producer.decode_forced(continuation, logits),
                          logits_path=str(logits),
                          logits_sha256=completed_logits_sha256(logits))
        if args.decode:
            record['decode'] = producer.decode(args.decode)
            steps = record['decode']['step_ms'][100:]
            record['warm_decode'] = dict(discarded_steps=min(100, args.decode), steps=len(steps),
                                         seconds=sum(steps) / 1000,
                                         tok_s=len(steps) * 1000 / sum(steps) if steps else None)
        record['status'] = 'PASS'
    except BaseException as error:
        record.update(status='FAIL', error=repr(error))
        raise
    finally:
        if producer is not None:
            record['cleanup'] = producer.close()
        if record['status'] == 'PASS' and record['flashinfer_requested']:
            text = trace.read_text(encoding='utf-8', errors='replace')
            actual = re.findall(r'NATIVEFP4_FLASHINFER_COMPLETE layer=(\d+) start=(\d+) count=(\d+) actual=(\d+) ', text)
            stored = re.findall(r'NATIVEFP4_STORED_COMPLETE layer=(\d+) start=(\d+) count=(\d+) actual=(\d+) ', text)
            if record['native_ffn_requested']:
                native=re.findall(r'NATIVEFP4_NATIVE_FFN_COMPLETE layer=(\d+) start=(\d+) count=(\d+) actual=1 products=3 routing_maps=1 input_gathers=1 runtime_weight_repack_bytes=0 routed_F32_bytes=0 epilogue_BF16=1',text)
                actual=[(*row,'3') for row in native];stored=actual
                record['native_ffn_completed_graphs']=len(native)
                record['runtime_weight_repack_bytes']=0
                if {int(row[0]) for row in native}!=set(range(40)):
                    record.update(status='FAIL',error='Native FFN witnesses do not cover all40 layers')
            record['flashinfer_completed_graphs'] = len(actual)
            record['flashinfer_completed_products'] = sum(int(row[3]) for row in actual)
            if not actual or actual != stored or any(int(row[3]) != (3 if record['native_ffn_requested'] or int(row[2]) > 4 else 0) for row in actual):
                record.update(status='FAIL', error='Missing or inconsistent completed FlashInfer W4A8 graph witnesses')
        if record['status'] == 'PASS' and record['decode_mma_requested'] and args.decode:
            layers = sorted(int(layer) for layer in re.findall(
                r'NATIVEFP4_DECODE_QUEUED layer=(\d+) calls=3 rows=1 offline_weights=1 runtime_weight_conversion=0 activation=fp4',
                trace.read_text(encoding='utf-8', errors='replace')))
            record['decode_mma_queued_layers'] = layers
            if layers != list(range(40)):
                record.update(status='FAIL', error='Missing native FP4 decode launch witnesses for all40 layers')
        if record['status']=='PASS' and record['dense_fp8_requested']:
            source='native_FP8' if record.get('native_dense_layout') else 'Q8_0'
            proxy='0' if record.get('native_dense_layout') else '1'
            actual=re.findall(r'NATIVEFP4_DENSE_FP8_COMPLETE layer=(\d+) start=(\d+) count=(\d+) actual=(\d+) source='+source+r' weight_scale=E8M0_32x32 activation=MXFP8 lossy_speed_proxy='+proxy,trace.read_text(encoding='utf-8',errors='replace'))
            record['dense_fp8_completed_products']=sum(int(row[3]) for row in actual)
            if {int(row[0]) for row in actual}!=set(range(40)) or not record['dense_fp8_completed_products']:
                record.update(status='FAIL',error='Missing FP8 dense completion witnesses for all40 layers')
        if record['status']=='PASS' and args.cuda_graphs:
            try:record.update(graph_witnesses(trace.read_text(encoding='utf-8',errors='replace'),max(1,record.get('native_ffn_completed_graphs',0))))
            except ValueError as error:record.update(status='FAIL',error=str(error))
        if record['status']=='PASS' and (record['asynchronous_layers_requested'] or record['layer_fusion_requested']):
            try:record.update(layer_witnesses(trace.read_text(encoding='utf-8',errors='replace'),record.get('native_ffn_completed_graphs',0),
                record['asynchronous_layers_requested'],record['layer_fusion_requested']))
            except ValueError as error:record.update(status='FAIL',error=str(error))
        args.output.write_text(json.dumps(record, indent=2, default=str), encoding='utf-8')
        if record['status'] == 'FAIL' and (record['asynchronous_layers_requested'] or record['layer_fusion_requested'] or args.cuda_graphs or record.get('dense_fp8_requested') or record.get('flashinfer_requested') or (
                record.get('decode_mma_requested') and 'decode_mma_queued_layers' in record)):
            raise RuntimeError(record['error'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--tokens', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--arm', required=True, choices=('q4', 'proxy'))
    p.add_argument('--bias', type=float, default=0)
    p.add_argument('--decode', type=int, default=0)
    p.add_argument('--continuation', type=Path)
    p.add_argument('--calibrate', type=Path)
    p.add_argument('--profile-stages', action='store_true')
    p.add_argument('--prefill-only',action='store_true',help='No decode/quality dial; optional one-token continuation saves only the untouched frontier')
    p.add_argument('--cuda-graphs',action='store_true',help='Explicit streamed layer capture/update/replay experiment; fail if unsupported')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if not 0 <= args.bias <= 1.36125 or args.decode < 0:
        p.error('Bias must be 0..1.36125 and decode count nonnegative')
    if args.output.exists():
        p.error('Refusing to overwrite an existing measurement')
    cfg = load(str(args.config))
    if args.prefill_only:
        if args.decode or args.calibrate or args.bias or (args.continuation and len(token_ids(args.continuation)[1])!=1):
            p.error('Prefill-only permits only bias0 and an optional single frontier probe')
        cfg['native_fp4_prefill_only']=True
        cfg['env_overrides'].update(LLAMADECODE_DS4_DECODE='0',LLAMADECODE_PREFILL_ROUTED='0')
    if args.calibrate:
        if args.bias or args.decode < 64 or args.calibrate.exists():
            p.error('Calibration needs bias zero, at least 64 decode tokens and a new output path')
        cfg['cache_affinity_spread_file'] = ''
        cfg['env_overrides']['DS4_CACHE_AFFINITY_CALIBRATE'] = str(args.calibrate.resolve())
    if args.arm == 'proxy':
        cfg['env_overrides']['LLAMA_DSV41_STORED_MXFP4']='1'
        if not args.cuda_graphs:cfg['env_overrides']['GGML_CUDA_DISABLE_GRAPHS']='1'
    if args.cuda_graphs:
        # Explicit CLI selection overrides the harness's inherited eager policy.
        cfg['env_overrides']['GGML_CUDA_DISABLE_GRAPHS']='0'
        cfg['env_overrides']['LLAMA_DSV41_CUDA_GRAPHS']='1'
    if cfg['env_overrides'].get('LLAMA_DSV41_LAYER_FUSION')=='2':
        if args.cuda_graphs:p.error('Select the pointer-table plan or the outer streamed-graph plan')
        # Explicit pointer-table selection overrides the inherited eager preset.
        # The native backend records its specialized graph plan in the trace.
        if cfg['env_overrides'].get('GGML_CUDA_DISABLE_GRAPHS')!='0':
            print('Explicit pointer-table plan enables CUDA graphs; replacing the inherited eager preset.',file=sys.stderr)
        cfg['env_overrides']['GGML_CUDA_DISABLE_GRAPHS']='0'
    if args.profile_stages:
        cfg['env_overrides'].update(LLAMA_CUDA_OP_PROFILE='1', LLAMA_CUDA_OP_PROFILE_NODES='1',
                                    LLAMA_CUDA_OP_PROFILE_DUMP_EACH='1', LLAMADECODE_TILE_TIMING='1')
    if args.worker:
        run(args, cfg)
    else:
        # OwnedProcess enforces the config's explicit host budget, physical floor,
        # paging stop, single-engine admission and shared GPU reservation.
        job = OwnedProcess(cfg, [sys.executable, '-B', str(Path(__file__).resolve()),
                                *sys.argv[1:], '--worker'], 'native-fp4-speed-proxy',
                           cwd=args.output.resolve().parent, gpu=True)
        receipt = job.wait()
        args.output.with_suffix('.guard.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        if receipt.get('exit_code') or receipt.get('reason'):
            raise RuntimeError('Guarded measurement failed; inspect the guard receipt and trace')


if __name__ == '__main__':
    main()
