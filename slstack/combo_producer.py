"""Original ULTRA session with the existing mode1 pipe and RAM checkpoint codec."""
from __future__ import annotations
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import secrets
import struct
import time
from .config import path, environment
from .journal import atomic_json, sha256
from .native_broker import vendor, model_receipt
from .native_profiles import profile
from .combo_contract import validate
from .combo_state import transport_checkpoint, payload_bytes


def canonical_contract(row):
    validate(row)
    return json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


class ComboProducer:
    def __init__(self,cfg,session,epoch,tokens,cancelled,*,base=None,runner=None):
        vendor()
        from studio.dsv41_session_contract import MODEL_SHA256
        from studio.dsv41_session_process import Producer
        from studio.dsv41_windows_runner import _Native
        self.session=session;self.epoch=epoch;self.process=None;self.closed=False;self.leases=ExitStack()
        if type(epoch) is not int or not 0<epoch<(1<<64):raise ValueError('Exact positive owner epoch required')
        if not tokens or len(tokens)>=cfg['context'] or any(type(t) is not int or not 0<=t<129280 for t in tokens):
            raise ValueError('Initial ULTRA append requires exact native token IDs within the admitted context')
        binary=path(cfg['llama_producer'])
        if sha256(binary)!=cfg['producer_sha256']:raise ValueError('ULTRA binary pin differs')
        self.directory=Path(cfg['runtime_root'])/('ultra-'+secrets.token_hex(8));self.directory.mkdir()
        native=runner or _Native();receipt=model_receipt(cfg)
        args=dict(profile('ultra')['arguments'])
        host_bytes=int(cfg.get('producer_host_budget_gib',8)*(1<<30))
        args.update({'--context':str(cfg['context']),'--capture-dspark':'0','--compressed-cache-bf16':'0',
            '--model':str(path(cfg['model'])),'--copy-receipt':str(receipt),'--build-sha':cfg['producer_sha256'],
            '--session-id':session,'--session-owner-epoch':str(epoch),'--job-host-bytes':str(host_bytes),
            '--receipt':str(self.directory/'producer.json'),'--carry':str(self.directory/'carry-not-created'),
            '--payload':str(self.directory/'payload-not-created')})
        token_path=self.directory/'append.tokens.le32'
        with token_path.open('xb') as output:
            for offset in range(0,len(tokens),8192):output.write(struct.pack('<'+'I'*len(tokens[offset:offset+8192]),*tokens[offset:offset+8192]))
        args['--tokens-le32']=str(token_path)
        bootstrap=None
        if base is not None:
            ram,contract=base;wire=canonical_contract(contract)
            if (contract['request_id'],contract['owner_epoch'],contract['context'])!=(session,epoch,cfg['context']):
                raise ValueError('Native base owner epoch/context differs')
            args['--native-base-contract-sha256']=hashlib.sha256(wire).hexdigest()
            args['--native-base-owner-epoch']=str(epoch)
            def bootstrap(write):
                with ram.lease():
                    write(wire+b'\n')
                    for offset in range(0,payload_bytes(contract['position']),65536):
                        write(ram.read(offset,min(65536,payload_bytes(contract['position'])-offset)))
        argv=[str(binary),*[str(v) for item in args.items() for v in item]]
        env=environment(cfg,'rtx-producer')
        env['PATH']=os.pathsep.join([str(binary.parent),*[str(path(p)) for p in cfg.get('producer_library_directories',[])],env.get('PATH','')])
        self.started=time.monotonic()
        try:
            self.leases.enter_context(native.directory(self.directory))
            for file in (binary,path(cfg['model']),receipt,token_path):self.leases.enter_context(native.read_lease(file))
            if native.gpu(0,cfg['gpu_uuid'])<int(args['--job-device-bytes'])+int(args['--gpu-reserve-bytes']):
                raise MemoryError('ULTRA GPU plan cannot fit; handoff refused')
            self.process=Producer(session,argv,env,self.directory,host_bytes=host_bytes,
                physical_floor=int(cfg['min_available_gib']*(1<<30)),operation_seconds=min(1140,cfg['request_timeout_s']),
                idle_seconds=cfg['lease_minutes']*60,native=native,cancelled=cancelled,native_mode=1,bootstrap=bootstrap,owner_epoch=epoch)
            self.checkpoint=self.process.ready['checkpoint']
            expected_position=(base[1]['position'] if base else 0)+len(tokens)
            if self.checkpoint['context']!=cfg['context'] or self.checkpoint['position']!=expected_position:
                raise ValueError('ULTRA initial frontier/context differs')
            self.prefill_seconds=time.monotonic()-self.started
            atomic_json(self.directory/'launch.json',dict(argv=argv,source_revision=cfg['producer_source_revision'],
                producer_sha256=cfg['producer_sha256'],owner_epoch=epoch,host_bytes=host_bytes,
                full_snapshot_bootstrap=base is None,weights_copied=False,profile='ultra',compressed_cache_bf16=False))
        except BaseException as exc:
            self.failure_cleanup=getattr(exc,'owned_cleanup',None)
            self.close()
            raise

    def command(self,operation,**extra):
        from studio.dsv41_session_contract import MODEL_SHA256
        return dict(session=self.session,model_sha256=MODEL_SHA256,sequence=self.process.sequence,operation=operation,**extra)

    def prefill(self,tokens):
        reply=self.process.request(self.command('prefill',base=self.checkpoint,tokens=tokens))
        self.checkpoint=reply['checkpoint'];return reply

    def export(self,ram):
        started=time.monotonic();cp=transport_checkpoint(self.checkpoint)
        with ram.lease():
            try:
                meta=self.process.begin_export(self.command('export_initial',base=cp))
                if meta['bytes']!=payload_bytes(cp['position']) or meta['bytes']>ram.capacity:
                    raise MemoryError('ULTRA exact payload exceeds admitted RAM checkpoint')
                offset=0
                while block:=self.process.read_export():ram.write(offset,block);offset+=len(block)
                self.process.end_export()
            except BaseException:
                self.process.abort_transfer();raise
        return dict(bytes=offset,seconds=time.monotonic()-started)

    def close(self):
        if self.closed:return self.receipt
        proof=self.process.close() if self.process else getattr(self,'failure_cleanup',None) or dict(creation_attempted=False)
        if self.process and not all(proof.get(k) for k in ('owned_job_empty','owned_process_signaled','owned_pid_gone')):
            raise RuntimeError('ULTRA owner retirement unconfirmed; decoder import refused')
        self.receipt=proof
        if self.process:(self.directory/'stderr.log').write_bytes(self.process.diagnostic)
        atomic_json(self.directory/'cleanup.json',proof)
        self.leases.close();self.closed=True;return proof
