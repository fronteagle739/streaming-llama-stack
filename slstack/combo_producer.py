"""Original ULTRA session with the existing mode1 pipe and RAM checkpoint codec."""
from __future__ import annotations
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import struct
import time
from .config import path, environment
from .journal import atomic_json, sha256
from .native_broker import vendor, model_receipt, model_digest
from .native_profiles import profile
from .combo_contract import validate
from .combo_state import transport_checkpoint, payload_bytes
from .combo_release import certified
from .combo_ram import protect_working_set,runtime_min_free_gib
from . import combo_abort


def canonical_contract(row):
    if 'local_state' in row:
        from .local_cache import validate_contract
        validate_contract(row)
    else:validate(row)
    return json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def startup_failure(error,directory):
    """Retain constructor stderr and its exact owned-cleanup attribution."""
    diagnostic=getattr(error,'owned_diagnostic',b'')
    if not isinstance(diagnostic,bytes) or not diagnostic:return error
    try:(Path(directory)/'stderr.log').write_bytes(diagnostic)
    except OSError as failure:
        error.add_note('Could not save producer startup diagnostic: '+repr(failure))
        return error
    if not isinstance(error,BrokenPipeError):return error
    tail=diagnostic.decode('utf-8',errors='replace')[-4000:].strip()
    reported=RuntimeError('ULTRA exited during base import: '+tail)
    reported.__dict__.update(error.__dict__)
    reported.original_error=repr(error)
    return reported



def ctx_cache(cfg):
    """LLAMADECODE producer context cache: (precision f32|bf16, growable) from the launcher config."""
    precision=cfg.get('ctx_cache_precision','f32');growable=cfg.get('ctx_cache_growable',False)
    if precision not in ('f32','bf16'):raise ValueError('ctx_cache_precision must be f32 or bf16')
    if type(growable) is not bool:raise ValueError('ctx_cache_growable must be true or false')
    if growable and cfg.get('producer_keep_alive'):raise ValueError('ctx_cache_growable is not covered by the switch arena (producer_keep_alive)')
    return precision,growable


def decode_host_budget(cfg, prefill_bytes):
    """A decode tier may reuse prefill RAM, inside a separately admitted ceiling.

    This ceiling does not enlarge the fold planner's prefill workspace. It is
    admitted before launch and watched by the same producer and aggregate guard.
    """
    value=cfg.get('producer_decode_host_budget_gib')
    if value is None:return prefill_bytes
    if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
        raise ValueError('producer_decode_host_budget_gib must be finite and positive')
    ceiling=max(prefill_bytes,int(value*(1<<30))//65536*65536)
    if ceiling+(.5*(1<<30))>cfg['host_budget_gib']*(1<<30):
        raise MemoryError('Decode host ceiling must leave 512 MiB for the controller inside host_budget_gib')
    return ceiling

LEGACY_SWEEP=49152
def carry_bank_bytes(rows):
    """Locked RAM carry bank for `rows` (llama_dsv41_prefill::carry_bank_bytes): 4 KiB header + 32-row pages."""
    return 4096+(rows+31)//32*2953216

def engram_prompt_bytes(rows):
    """Whole-prompt Engram row cache admission (llama_dsv41_prefill::engram_prompt_bytes)."""
    return (4<<20)+rows*48*268

FOLD_CAP=61440   # owner, measured: fold wins at 54,037 and 60,176, loses at 93,668

def fold_sweeps(prompt_tokens,sweep,capacity,fold_cap=None,align=4096):
    """Owner fold rule (llama_dsv41_prefill::fold): primary sweeps of `sweep` tokens; the remainder
    r = n - k*sweep is folded into the last one, last = sweep + r; n <= sweep is one sweep.
    last > fold_cap: the remainder is its own sweep (tail). last > capacity but <= fold_cap
    (RAM-limited): rebalance `last` into m near-equal passes (m from ceil(last / capacity floored to
    align); every pass but the last is ceil(last/m) rounded up to align, the last takes the rest; m
    grows while a pass exceeds capacity), never a tiny tail. Returns (sweep sizes, tail, rebalance)."""
    n=int(prompt_tokens)
    if n<=sweep:return [n],False,False
    k,r=divmod(n,sweep)
    if not r:return [sweep]*k,False,False
    last=sweep+r
    if fold_cap and last>fold_cap:return [sweep]*k+[r],True,False
    if last<=capacity:return [sweep]*(k-1)+[last],False,False
    grid=capacity//align*align
    if not grid:raise ValueError('Fold capacity %d is below one %d-token pass'%(capacity,align))
    m=-(-last//grid)
    while True:
        p=-(-(-(-last//m))//align)*align
        if p<=capacity and p*(m-1)<last:return [sweep]*(k-1)+[p]*(m-1)+[last-p*(m-1)],False,True
        m+=1

def prefill_plan(cfg,prompt_tokens,available=None):
    """LLAMADECODE single sweep (owner: one prompt = one weight sweep). The producer's capacity C
    (--prefill-chunk-tokens) is sized from the prompt: C = max(49152, prompt), raising the producer
    host budget by exactly the extra carry bank + Engram bytes, up to producer_host_max_gib (default:
    producer_host_budget_gib, i.e. no raise and today's 49152). A prompt over the largest C that fits
    is split by the producer into near-equal sweeps. Returns C, the raised budget and the deltas.
    `available` (physical RAM measured at launch) also caps the budget at available minus
    producer_host_reserve_gib (default 0.5 GiB of headroom for Studio/daemon/OS growth), so the working set plus everything
    else fits physical RAM and stays resident (hard-minimum working set); the budget is
    page-rounded (Windows keeps a hard-minimum working set only in whole pages).
    producer_prefill_sweep_tokens S (absent = the above): the owner fold policy instead; S-token
    sweeps with the remainder folded into the last (fold_sweeps), C = the largest sweep (at least
    49152), the same host raise for C; when the folded sweep cannot fit the cap, or exceeds the fold
    cap producer_prefill_fold_cap_tokens (default 61440), the remainder is its own sweep."""
    base=int(cfg.get('producer_host_budget_gib',8)*(1<<30))
    top=int(cfg.get('producer_host_max_gib',cfg.get('producer_host_budget_gib',8))*(1<<30))
    if top<base:raise ValueError('producer_host_max_gib is below producer_host_budget_gib')
    capped_by='producer_host_max_gib'
    if available is not None:
        # Owner 2026-10-25 ("every run makes you freeze like the computer's run out of RAM"): locked producer
        # pages cannot be paged out, so admission keeps a DESKTOP RESERVE free for Windows, Claude Code and the
        # desktop: the producer's raised budget is at most available - producer_desktop_reserve_gib (per
        # machine, default 0.5 GiB). The separate runtime_min_free_gib stop defaults to 0.5 GiB.
        reserve=max(float(cfg.get('producer_host_reserve_gib',.5)),float(cfg.get('producer_desktop_reserve_gib',.5)))
        room=int(available)-int(reserve*(1<<30))
        if room<top:top=max(base,room);capped_by='desktop_reserve' if reserve>float(cfg.get('producer_host_reserve_gib',.5)) else 'available_ram'
    extra=lambda c:(carry_bank_bytes(c)-carry_bank_bytes(LEGACY_SWEEP),engram_prompt_bytes(c)-engram_prompt_bytes(LEGACY_SWEEP))
    limit=min(1<<20,int(cfg['context'])-1)
    want=min(max(LEGACY_SWEEP,int(prompt_tokens)),limit)
    fold=cfg.get('producer_prefill_sweep_tokens')
    if fold is not None:
        return fold_plan(cfg,prompt_tokens,available,fold,base,top,capped_by,extra,limit)
    if base+sum(extra(want))>top:   # the largest whole-page capacity under the cap
        lo,hi=LEGACY_SWEEP,want
        while lo<hi:
            mid=(lo+hi+1)//2
            if base+sum(extra(mid))<=top:lo=mid
            else:hi=mid-1
        want=lo
        # Splitting anyway: reserve only the near-equal chunk (less locked RAM, more margin).
        from studio.dsv41_session_contract import split_prefill
        want=max(LEGACY_SWEEP,split_prefill(int(prompt_tokens),want))
    carry,engram=extra(want)
    host=base+carry+engram
    if host!=base:host=-(-host//65536)*65536   # whole pages (64 KiB) for the hard-minimum working set
    single=int(prompt_tokens)<=want
    return dict(capacity=want,host_bytes=host,carry_delta=carry,engram_delta=engram,base_host_bytes=base,max_host_bytes=top,
                available_bytes=available,capped_by=capped_by,prompt_tokens=int(prompt_tokens),single_sweep=single,
                reason=None if single else 'one sweep of %d tokens needs %d host bytes; the cap is %d (%s); near-equal sweeps of %d'%(
                    int(prompt_tokens),base+sum(extra(min(max(LEGACY_SWEEP,int(prompt_tokens)),limit))),top,capped_by,want))

def fold_plan(cfg,prompt_tokens,available,sweep,base,top,capped_by,extra,limit):
    """prefill_plan under producer_prefill_sweep_tokens (the owner fold policy)."""
    if type(sweep) is not int or sweep<=0 or sweep%4096 or sweep>limit:
        raise ValueError('producer_prefill_sweep_tokens must be a positive multiple of 4096 within the context')
    cap=limit
    if base+sum(extra(cap))>top:   # the largest capacity under the host cap (same search as prefill_plan)
        lo,hi=LEGACY_SWEEP,cap
        while lo<hi:
            mid=(lo+hi+1)//2
            if base+sum(extra(mid))<=top:lo=mid
            else:hi=mid-1
        cap=lo
    fold_cap=cfg.get('producer_prefill_fold_cap_tokens',FOLD_CAP)
    if type(fold_cap) is not int or fold_cap<=0:raise ValueError('producer_prefill_fold_cap_tokens must be a positive token count')
    n=int(prompt_tokens)
    if n>sweep and sweep>cap:
        raise ValueError('producer_prefill_sweep_tokens %d exceeds the largest capacity %d under the cap (%s)'%(sweep,cap,capped_by))
    sweeps,tail,rebalance=fold_sweeps(n,sweep,cap,fold_cap)
    want=max(LEGACY_SWEEP,max(sweeps))
    carry,engram=extra(want)
    host=base+carry+engram
    if host!=base:host=-(-host//65536)*65536   # whole pages (64 KiB) for the hard-minimum working set
    return dict(capacity=want,host_bytes=host,carry_delta=carry,engram_delta=engram,base_host_bytes=base,max_host_bytes=top,
                available_bytes=available,capped_by=capped_by,prompt_tokens=n,single_sweep=len(sweeps)==1,
                policy='fold',sweep_tokens=sweep,sweeps=sweeps,fold_tail=tail,fold_cap=fold_cap,rebalance=rebalance,
                reason='fold: last sweep %d + %d = %d exceeds the fold cap %d; the %d-token remainder is its own sweep'%(
                    sweep,sweeps[-1],sweep+sweeps[-1],fold_cap,sweeps[-1]) if tail else
                    'fold: last sweep %d exceeds the largest capacity %d (%s); rebalanced into %s'%(
                    sum(sweeps[n//sweep-1:]),cap,capped_by,'+'.join(map(str,sweeps[n//sweep-1:]))) if rebalance else None)

def stall_seconds(cfg):
    """No-progress window for a producer operation. A long operation (a 262K-token prefill, a long
    decode) lives while the producer shows progress; request_timeout_s only bounds silence."""
    value=float(cfg.get('single_engine_stall_seconds',600))
    if not 30<=value<=3600:raise ValueError('single_engine_stall_seconds must be 30-3600')
    return value

def yoco_keep(cfg):
    keep=cfg.get('yoco_keep',0)
    if type(keep) is not int or keep not in (0,140):
        raise ValueError('yoco_keep must be 0 or 140')
    if keep and cfg.get('serving_mode',cfg.get('serving'))!='single_engine':
        # The paired route uses native_broker and negotiates mode3 state v1.
        # ComboProducer's local snapshot/restore contract is a different route.
        raise ValueError('ComboProducer YOCO requires local single_engine; paired YOCO uses the negotiated native broker')
    return keep


def prefill_arguments(cfg, count, available, *, export_state=False, techniques=None):
    """One arithmetic/storage/fold plan for single-engine and paired prefill.

    Mode3 export additionally captures the 7.5MiB DSpark ring required by the
    Mac importer. It does not select another arithmetic or storage profile.
    """
    selected = dict(cfg)
    for key, value in dict(producer_host_budget_gib=6.25, producer_host_max_gib=7.75,
                           producer_prefill_sweep_tokens=49152,
                           producer_prefill_fold_cap_tokens=61440).items():
        selected.setdefault(key, value)
    arguments = dict(profile(cfg.get('producer_profile', 'ultra'), techniques if techniques is not None else cfg.get('native_techniques'))['arguments'])
    plan = prefill_plan(selected, count, available)
    arguments['--prefill-chunk-tokens'] = str(plan['capacity'])
    arguments['--job-host-bytes'] = str(plan['host_bytes'])
    for key, delta in (('--carry-host-bytes', plan['carry_delta']),
                       ('--provider-host-bytes', plan['engram_delta']),
                       ('--engram-prefetch-bytes', plan['engram_delta'])):
        arguments[key] = str(int(arguments[key]) + delta)
    arguments['--prefill-sweep-tokens'] = str(plan['sweep_tokens'])
    arguments['--prefill-fold-cap-tokens'] = str(plan['fold_cap'])
    precision, growable = ctx_cache(cfg)
    arguments.update({'--context': str(cfg['context']),
                      '--capture-dspark': '1' if export_state else '0',
                      '--compressed-cache-bf16': '1' if precision == 'bf16' else '0'})
    if precision != 'f32' or growable:
        arguments.update({'--ctx-cache-precision': precision, '--ctx-cache-growable': str(int(growable))})
    if growable:
        # Same eight owner tensors as llama-dsv41-ctx-cache-plan.h. The
        # qualified RTX VMM granularity is 2 MiB. Its padding makes the exact
        # mode3 1,000,001-position window need 12 KiB more than the old bill.
        # Reassign that padding from graph headroom; keep the device cap fixed.
        granularity=2<<20;context=int(cfg['context']);element=2 if precision=='bf16' else 4
        fixed=40*128*512*4+3*2*512*4
        mapped=sum(-(-(((context+1)//2 if owner<3 else context)*width*element)//granularity)*granularity
                   for owner in range(4) for width in (512,128))
        extra=max(0,fixed+mapped-int(arguments['--cache-bytes']))
        arguments['--cache-bytes']=str(int(arguments['--cache-bytes'])+extra)
        arguments['--graph-bytes']=str(int(arguments['--graph-bytes'])-extra)
        if int(arguments['--graph-bytes'])<=0:raise MemoryError('Context cache leaves no admitted graph budget')
    if cfg.get('producer_fixed_direct_open', True) and arguments.get('--fixed-resident-plan') != '1':
        arguments.update({'--warmup-one-token': '0', '--fixed-direct-open': '1'})
    keep = cfg.get('yoco_keep', 0)
    if type(keep) is not int or keep not in (0, 140):raise ValueError('yoco_keep must be 0 or 140')
    if keep:
        arguments['--yoco-keep'] = str(keep)
        if export_state:arguments['--yoco-remote-state-version'] = '1'
    return arguments, plan


class ComboProducer:
    def __init__(self,cfg,session,epoch,tokens,cancelled,*,base=None,runner=None,on_start=None):
        combo_abort.check('producer launch')
        keep=yoco_keep(cfg)
        if base is not None and 'local_state' in base[1]:
            from .local_cache import validate_contract
            validate_contract(base[1])
            local=base[1]['local_state']
            if (local['yoco_keep'],local['cache_precision'],local['producer_sha256'])!=(keep,cfg.get('ctx_cache_precision','f32'),cfg['producer_sha256']):
                raise ValueError('Local snapshot producer/YOCO/cache preset differs')
        elif keep and base is not None:raise ValueError('YOCO requires a local snapshot with YOCO provenance')
        runtime_floor=runtime_min_free_gib(cfg)
        vendor()
        from studio.dsv41_session_process import Producer
        from studio.dsv41_windows_runner import _Native
        self.session=session;self.epoch=epoch;self.process=None;self.closed=False;self.leases=ExitStack()
        self.loading_process=None
        if on_start:on_start(self)
        self.cfg=cfg;self.shutdown_path='terminate';self.shutdown_seconds=0.0;self.shutdown_detail=None;self._exit_code=None
        self.creation_attempted=False
        if type(epoch) is not int or not 0<epoch<(1<<64):raise ValueError('Exact positive owner epoch required')
        if not tokens or len(tokens)>=cfg['context'] or any(type(t) is not int or not 0<=t<129280 for t in tokens):
            raise ValueError('Initial ULTRA append requires exact native token IDs within the admitted context')
        binary=path(cfg['llama_producer'])
        if sha256(binary)!=cfg['producer_sha256']:raise ValueError('ULTRA binary pin differs')
        self.directory=Path(cfg['runtime_root'])/('ultra-'+secrets.token_hex(8));self.directory.mkdir()
        native=runner or _Native();receipt=model_receipt(cfg)
        self.model_sha256=model_digest(cfg,receipt)
        tier=cfg.get('producer_profile','ultra')
        if tier not in ('accurate','fast','ultra'):raise ValueError('Combo producer profile must be accurate, fast or ultra (original weights; no HYPER)')
        import psutil
        args,plan=prefill_arguments(cfg,len(tokens),psutil.virtual_memory().available)
        self.prefill_plan=plan;host_bytes=decode_host_budget(cfg,plan['host_bytes'])
        if plan.get('policy')=='fold':   # owner fold policy: S-token sweeps, remainder folded into the last
            args['--prefill-sweep-tokens']=str(plan['sweep_tokens']);args['--prefill-fold-cap-tokens']=str(plan['fold_cap'])
            print('LLAMADECODE_PREFILL_FOLD sweep_tokens=%s primary_sweep_tokens=%d capacity=%d fold_cap=%d rebalance=%d%s'%(','.join(map(str,plan['sweeps'])),plan['sweep_tokens'],plan['capacity'],plan['fold_cap'],int(plan['rebalance']),
                ' reason='+plan['reason'] if plan['rebalance'] else ''),flush=True)
            if plan['fold_tail']:print('LLAMADECODE_PREFILL_FOLD_TAIL '+plan['reason'],flush=True)
        elif not plan['single_sweep']:print('LLAMADECODE_PREFILL_SINGLE_SWEEP_REFUSED '+plan['reason'],flush=True)
        print('LLAMADECODE_PREFILL_BUDGET '+json.dumps(plan),flush=True)
        precision,growable=ctx_cache(cfg)
        args.update({'--context':str(cfg['context']),'--capture-dspark':'0','--compressed-cache-bf16':'1' if precision=='bf16' else '0',
            '--model':str(path(cfg['model'])),'--copy-receipt':str(receipt),'--build-sha':cfg['producer_sha256'],
            '--session-id':session,'--session-owner-epoch':str(epoch),'--job-host-bytes':str(host_bytes),
            '--runtime-min-free-gib':str(runtime_floor),
            '--receipt':str(self.directory/'producer.json'),'--carry':str(self.directory/'carry-not-created'),
            '--payload':str(self.directory/'payload-not-created')})
        if precision!='f32' or growable:   # LLAMADECODE context cache flags (absent = today's f32 fixed argv)
            args['--ctx-cache-precision']=precision;args['--ctx-cache-growable']=str(int(growable))
        if cfg.get('producer_keep_alive'):args['--switch-arena']='1'   # DS4COMBO fast switch: suspendable device memory
        token_path=self.directory/'append.tokens.le32'
        with token_path.open('xb') as output:
            for offset in range(0,len(tokens),8192):output.write(struct.pack('<'+'I'*len(tokens[offset:offset+8192]),*tokens[offset:offset+8192]))
        args['--tokens-le32']=str(token_path)
        if cfg.get('_vision_spans'):
            descriptor=self.directory/'vision-spans.json';atomic_json(descriptor,cfg['_vision_spans'])
            args['--vision-spans-file']=str(descriptor)
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
                    size=payload_bytes(contract['position'],local='local_state' in contract)
                    for offset in range(0,size,65536):
                        write(ram.read(offset,min(65536,size-offset)))
        argv=[str(binary),*[str(v) for item in args.items() for v in item]]
        env=environment(cfg,'rtx-producer')
        if not env.get('LLAMADECODE_ABORT_FILE'):env['LLAMADECODE_ABORT_FILE']=str(self.directory/'guard.abort')
        if cfg.get('cache_affinity_spread_file') and not env.get('LLAMADECODE_DS4_SPREAD'):
            env['LLAMADECODE_DS4_SPREAD']=str(path(cfg['cache_affinity_spread_file']))   # B5 calibration (native's file)
        self.cancel_path=self.directory/'decode.cancel'   # LLAMADECODE decode cancel (see cancel_decode)
        env['LLAMADECODE_DECODE_CANCEL_FILE']=str(self.cancel_path)
        env['PATH']=os.pathsep.join([str(binary.parent),*[str(path(p)) for p in cfg.get('producer_library_directories',[])],env.get('PATH','')])
        self.started=time.monotonic()
        try:
            self.leases.enter_context(native.directory(self.directory))
            for file in (binary,path(cfg['model']),receipt,token_path):self.leases.enter_context(native.read_lease(file))
            if cfg.get('_vision_spans'):
                self.leases.enter_context(native.read_lease(descriptor))
                for span in cfg['_vision_spans']:self.leases.enter_context(native.read_lease(Path(span['file'])))
            if native.gpu(0,cfg['gpu_uuid'])<int(args['--job-device-bytes'])+int(args['--gpu-reserve-bytes']):
                raise MemoryError('ULTRA GPU plan cannot fit; handoff refused')
            self.creation_attempted=True
            owner=self
            class PublishingProducer(Producer):
                def _initialize(process,*args,**kwargs):
                    owner.loading_process=process
                    try:return super()._initialize(*args,**kwargs)
                    finally:owner.loading_process=None
            def admitted(job):
                # A lost controller must leave a GPU owner recoverable. Memory
                # and paging stops use its abort file and retirement barrier.
                job.release_kill_on_close()
                return protect_working_set(job,host_bytes,int(cfg['min_available_gib']*(1<<30)),
                    int(cfg['host_budget_gib']*(1<<30)),receipt=self.directory/'resident-admission.json')
            self.process=PublishingProducer(session,argv,env,self.directory,host_bytes=host_bytes,
                runtime_min_free_gib=runtime_floor,model_sha256=self.model_sha256,
                physical_floor=int(cfg['min_available_gib']*(1<<30)),operation_seconds=min(1140,cfg['request_timeout_s']),stall_seconds=stall_seconds(cfg),
                idle_seconds=cfg['lease_minutes']*60,native=native,cancelled=cancelled,native_mode=1,bootstrap=bootstrap,owner_epoch=epoch,
                after_resume=admitted)
            if keep and self.process.ready.get('yoco_keep')!=keep:raise ValueError('Producer YOCO mode identity differs')
            self.checkpoint=self.process.ready['checkpoint']
            expected_position=(base[1]['position'] if base else 0)+len(tokens)
            if self.checkpoint['context']!=cfg['context'] or self.checkpoint['position']!=expected_position:
                raise ValueError('ULTRA initial frontier/context differs')
            self.prefill_seconds=time.monotonic()-self.started
            atomic_json(self.directory/'launch.json',dict(argv=argv,source_revision=cfg['producer_source_revision'],
                producer_sha256=cfg['producer_sha256'],owner_epoch=epoch,host_bytes=host_bytes,
                resident_admission=getattr(self.process,'resume_admission',None),
                full_snapshot_bootstrap=base is None,weights_copied=False,profile=tier,compressed_cache_bf16=precision=='bf16',
                ctx_cache_precision=precision,ctx_cache_growable=growable,prefill_plan=plan,yoco_keep=keep))
        except BaseException as exc:
            self.failure_cleanup=getattr(exc,'owned_cleanup',None)
            if self.process is not None:
                try:self.failure_cleanup=self.process.close()
                except BaseException as cleanup:exc.owned_cleanup_error=repr(cleanup)
            if self.failure_cleanup is not None:exc.owned_cleanup=self.failure_cleanup
            if not self.creation_attempted or certified(self.failure_cleanup):
                try:self.close()
                except BaseException as cleanup:exc.owned_cleanup_error=repr(cleanup)
            else:
                exc.owned_cleanup=self.failure_cleanup or dict(creation_attempted=True)
                exc.owned_runtime=self
            reported=startup_failure(exc,self.directory)
            if reported is not exc:raise reported from exc
            raise

    def command(self,operation,**extra):
        if operation!='close':combo_abort.check('producer '+operation)
        return dict(session=self.session,model_sha256=self.model_sha256,sequence=self.process.sequence,operation=operation,**extra)

    # --- DS4COMBO fast switch: keep-alive producer (suspend / resume) ---------
    def suspend(self):
        """After the native imported this frontier: acknowledge it, then give back
        VRAM/DS heaps/carry RAM while the process, CUDA context and model lease stay."""
        started=time.monotonic();cp=transport_checkpoint(self.checkpoint)
        self.process.request(self.command('acknowledge',base=cp))
        reply=self.process.suspend(cp);self.suspended=True
        return dict(reply=reply,seconds=time.monotonic()-started)

    def resume(self,base,tokens):
        """Map device memory again, stream the new native base on the session pipe,
        and prefill `tokens` on it. Same contract/payload writer as the launch bootstrap."""
        if not getattr(self,'suspended',False):raise RuntimeError('Producer resume requires a suspended keep-alive producer')
        ram,contract=base;wire=canonical_contract(contract)
        if (contract['request_id'],contract['owner_epoch'],contract['context'])!=(self.session,self.epoch,self.checkpoint['context']):
            raise ValueError('Native base owner epoch/context differs')
        def bootstrap(write):
            with ram.lease():
                write(wire+b'\n')
                for offset in range(0,payload_bytes(contract['position']),65536):
                    write(ram.read(offset,min(65536,payload_bytes(contract['position'])-offset)))
        started=time.monotonic()
        reply=self.process.resume(hashlib.sha256(wire).hexdigest(),tokens,bootstrap)
        self.checkpoint=reply['checkpoint'];self.suspended=False
        if self.checkpoint['position']!=contract['position']+len(tokens):raise ValueError('ULTRA resumed frontier differs')
        self.prefill_seconds=time.monotonic()-started
        return dict(reply=reply,seconds=self.prefill_seconds)
    # --- end DS4COMBO fast switch ---------------------------------------------

    def prefill(self,tokens):
        reply=self.process.request(self.command('prefill',base=self.checkpoint,tokens=tokens))
        self.checkpoint=reply['checkpoint'];return reply

    def decode(self,max_tokens,stop_tokens=()):
        """Native decode inside this producer using its configured sampler. The
        returned last token (stop, or sampled at max_tokens) is not evaluated."""
        reply=self.process.request(self.command('decode',base=self.checkpoint,max_tokens=int(max_tokens),stop_tokens=[int(t) for t in stop_tokens]))
        self.checkpoint=reply['checkpoint'];return reply

    def cancel_decode(self):
        """From another thread while decode() runs: the producer stops at its next
        token boundary, exactly as at max_tokens (reply cancelled=true; the last
        returned token is not evaluated). It clears the file at every decode start."""
        try:self.cancel_path.touch()
        except OSError:pass

    def quality(self,value):
        """LLAMADECODE B5: the live quality-speed setting (0 = max quality, 1.36125 = max speed) for the
        in-process ds4 decode; same scale as ds4-win /v1/quality-speed. Applies from the next decode."""
        return self.process.request(self.command('quality_speed',value=float(value)))

    def generation(self, settings):
        if self.process.ready.get('generation_control_version') != 1:
            raise ValueError('This producer needs the live generation-control build; greedy fallback is refused')
        return self.process.request(self.command('generation_start', settings=settings))

    def decode_forced(self,tokens,logits_path):
        """LLAMADECODE gate: teacher-forced 1-row decode; writes one full F32 logits row per token."""
        reply=self.process.request(self.command('decode_forced',base=self.checkpoint,tokens=[int(t) for t in tokens],logits_path=str(logits_path)))
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

    def graceful_exit(self):
        """Ask the producer to leave its session loop and release its sweep (DS
        queues and heaps, imported D3D12 buffers, locked carry RAM, ggml device
        memory) before it exits. Graceful only with exit code 0 and the
        completion line on stderr; anything else falls back to the Job
        terminate, and the fallback is logged."""
        p=self.process;job=getattr(p,'job',None)
        if job is None or job.poll() is not None or p.closed or p.failure is not None or p.lock.locked():
            self.shutdown_path='terminate: producer not idle or already exited';return
        started=time.monotonic();exited=None
        try:
            p.request(self.command('close'))
            # The budget starts once the reply has arrived.
            from .shutdown import wait_for_exit
            exited=wait_for_exit(job,float(self.cfg.get('producer_graceful_shutdown_seconds',60)),
                diagnostic=lambda:len(p.diagnostic))
            self.shutdown_path='graceful-pending' if exited is not None else 'graceful-timeout-terminate'
        except BaseException as exc:
            self.shutdown_path='graceful-failed-terminate: '+repr(exc)[:200]
        self.shutdown_seconds=time.monotonic()-started;self._exit_code=exited

    def _graceful_verdict(self):
        # Read after Producer.close(), so stderr has been fully drained.
        if self.shutdown_path!='graceful-pending':return
        done=re.search(rb'DS4COMBO_PRODUCER_GRACEFUL_SHUTDOWN_COMPLETE [^\r\n]*',bytes(self.process.diagnostic or b''))
        self.shutdown_path='graceful' if self._exit_code==0 and done else 'graceful-crashed exit='+str(self._exit_code)
        if done:self.shutdown_detail=done.group(0).decode(errors='replace')

    def close(self):
        if self.closed:return self.receipt
        if self.process:self.graceful_exit()
        # The owned runner repeats cancellation and waits while retirement advances;
        # only a full quiet grace allows its loudly logged force fallback.
        job=getattr(self.process,'job',None) if self.process else None
        if job is not None and job.poll() is None:
            print('DS4COMBO producer graceful retire did not finish ('+str(self.shutdown_path)+'); requesting cooperative cancellation and waiting for retirement',flush=True)
        try:
            proof=self.process.close() if self.process else getattr(self,'failure_cleanup',None) or dict(creation_attempted=False)
        except RuntimeError as exc:
            if 'STUCK PROCESS' in str(exc):
                from .combo_native import record_stuck
                record_stuck(self.cfg,self.directory,self.process,exc)
            raise
        if self.process:
            self._graceful_verdict()
            if self.shutdown_path!='graceful':
                print('DS4COMBO producer shutdown fell back: '+self.shutdown_path,flush=True)
            proof=dict(proof,producer_shutdown_path=self.shutdown_path,producer_shutdown_seconds=self.shutdown_seconds,producer_shutdown_detail=self.shutdown_detail)
        if self.creation_attempted and not certified(proof):
            raise RuntimeError('ULTRA owner retirement unconfirmed; decoder import refused')
        self.receipt=proof
        try:
            if self.process:(self.directory/'stderr.log').write_bytes(self.process.diagnostic)
            atomic_json(self.directory/'cleanup.json',proof)
        finally:self.leases.close();self.closed=True
        return proof
