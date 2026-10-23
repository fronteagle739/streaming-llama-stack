"""Attest the completed native fixture without publishing its environment."""
import argparse
import json
from pathlib import Path
from .journal import atomic_json,sha256


def bind_native(cfg):
    root=Path(cfg['runtime_root'])
    proof=json.loads((root/'native-parity.json').read_text())
    guard=json.loads((root/'native-parity.receipt.json').read_text())
    spec=json.loads((root/'native-parity.spec.json').read_text())
    if proof.get('status')!='PASS' or any(proof.get(key)!=384 for key in ('roundtrip_byte_exact','rewind_byte_exact')):
        raise ValueError('Native GPU proof is incomplete')
    if guard.get('exit_code')!=0 or guard.get('reason') is not None:raise ValueError('Native guardian did not pass')
    pins=spec['pins']
    if any(pins.get(key)!=cfg[key] for key in ('native_sha256','producer_sha256')):
        raise ValueError('Native gate engine pins differ')
    command=spec['command']
    if command[1]!=cfg['model'] or sha256(command[0])!=pins['fixture_sha256']:
        raise ValueError('Native fixture or original model path changed')
    if sha256(cfg['ds4_command'][0])!=cfg['native_sha256'] or sha256(cfg['llama_producer'])!=cfg['producer_sha256']:
        raise ValueError('Qualified engine binary changed')
    receipt=json.loads(Path(cfg['model_receipt']).read_text())['receipts'][Path(cfg['model']).name]
    stat=Path(cfg['model']).stat()
    identity=dict(size=stat.st_size,dev=stat.st_dev,ino=stat.st_ino,mtime_ns=stat.st_mtime_ns)
    if receipt['final_identity']!=identity or receipt['sha256']!=cfg['combo_identity']['model_sha256']:
        raise ValueError('Original model physical identity changed after the native gate')
    row=dict(status='PASS',native_proof_sha256=sha256(root/'native-parity.json'),
        native_guardian_sha256=sha256(root/'native-parity.receipt.json'),
        native_specification_sha256=sha256(root/'native-parity.spec.json'),
        fixture_sha256=pins['fixture_sha256'],native_sha256=cfg['native_sha256'],producer_sha256=cfg['producer_sha256'],
        identity=cfg['combo_identity'],provider_schema=cfg['provider_schema'],native_prefill_chunk=cfg['native_prefill_chunk'],
        decoder_baseline_revision='035c04d2b17024132a9ee37932f7e95b72b07822',
        scope='Real CUDA native core fixture:384 complete vectors and greedy IDs, full reopen/restore and saved-frontier rewind; ULTRA comparison is separate',
        raw_environment_published=False)
    target=root/'native-parity-binding.json'
    if target.exists() and json.loads(target.read_text())!=row:
        raise ValueError('Existing native attestation differs; preserve and investigate')
    atomic_json(target,row);return row


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    row=bind_native(json.loads(Path(args.config).read_text()));print(json.dumps(dict(status=row['status'],fixture_sha256=row['fixture_sha256'])))
