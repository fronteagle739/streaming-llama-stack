"""One GPU, one owner epoch: native -> locked RAM -> ULTRA -> native.

The outer request guardian owns the lease and aggregate process-tree budget.
This class never falls back after a failed transition. A poisoned epoch must
be explicitly replaced by the client; it cannot publish its retained bytes.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import secrets
import struct
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from .combo_native import Native
from .combo_producer import ComboProducer
from .combo_ram import LockedRam,wait_for_host_capacity
from .combo_state import contract_from_checkpoint,verify_payload,payload_bytes
from .combo_release import release_barrier,certified,suspend_barrier
from .combo_crossover import qualified_threshold
from .native_broker import vendor


def prefix_digest(tokens,position):
    digest=hashlib.sha256()
    for off in range(0,position,8192):
        part=tokens[off:min(position,off+8192)];digest.update(struct.pack('<'+'I'*len(part),*part))
    return digest.hexdigest()


def changed_span_tokens(tokens,blank):
    """Exact changed span between two renders from the native tokenizer."""
    left=0
    while left<min(len(tokens),len(blank)) and tokens[left]==blank[left]:left+=1
    right=0
    while right<min(len(tokens)-left,len(blank)-left) and tokens[-1-right]==blank[-1-right]:right+=1
    return len(tokens)-left-right


def render_body(body):
    vendor()
    from studio.reasoning_history import apply
    result=deepcopy(body)
    policy=result.pop('include_previous_reasoning',True)
    apply(result,{'include_previous_reasoning':policy})
    for message in result.get('messages',[]):
        content=message.get('content')
        if isinstance(content,list) and any(not isinstance(part,dict) or part.get('type') not in ('text','input_text') for part in content):
            raise ValueError('The combined preset requires text documents; multimodal state is outside its qualified contract')
    if any(key in result for key in ('images','image','image_url','audio','input_audio')):
        raise ValueError('The combined preset requires text documents')
    return result


class ComboSession:
    def __init__(self,cfg,*,device=None,qualification_route=None):
        vendor()
        from studio.dsv41_windows_runner import _Native
        self.cfg=deepcopy(cfg);self.device=device or _Native()
        self.session=secrets.token_hex(16);self.epoch=secrets.randbits(63) or 1
        self.generation=0;self.state=None;self.contract=None;self.payload=None
        self.journal=None;self.anchor=None;self.current=None;self.poison=None
        self.failed_owner=None
        self.provenance='unbiased';self.anchor_provenance=None
        self.cancel=threading.Event();self.lock=threading.Lock();self.quality_setting=None
        if qualification_route not in (None,'native','ultra'):raise ValueError('Unknown explicit qualification route')
        self.qualification_route=qualification_route
        self.threshold=qualified_threshold(cfg) if qualification_route is None else None
        # expert_cache_restore (default true): carry the native expert cache across relaunches.
        self.expert_list=(Path(cfg['runtime_root'])/('expert-cache-'+self.session+'.txt')
                          if cfg.get('expert_cache_restore',True) else None)
        # DS4COMBO fast switch: a suspended ULTRA producer kept across handoffs.
        self.kept_producer=None

    def _ram(self,size):
        return LockedRam(size,session=self.session,host_budget_gib=self.cfg['host_budget_gib'],min_available_gib=self.cfg['min_available_gib'])

    def _checkpoint(self,owner,cp):
        started=time.monotonic()
        self.generation+=1
        contract=contract_from_checkpoint(cp,self.cfg['combo_identity'],self.session,self.epoch,self.generation,self.provenance)
        # Reuse an admitted larger section. Never allocate a second full state.
        needed=payload_bytes(cp['position'])
        if self.state is not None and self.state.capacity<needed:self.state.close();self.state=None
        if self.state is None:self.state=self._ram(needed)
        export_start=time.monotonic();exported=owner.export(self.state)
        export_seconds=time.monotonic()-export_start;verify_start=time.monotonic()
        payload=verify_payload(self.state,contract,cp.get('payload_sha256'))
        self.contract,self.payload=contract,payload
        return dict(export=exported,payload=payload,export_seconds=export_seconds,
                    verification_seconds=time.monotonic()-verify_start,seconds=time.monotonic()-started,
                    locked_capacity_bytes=self.state.capacity)

    def quality(self,value=None):
        owner=self.current
        if value is not None:self.quality_setting=deepcopy(value)
        if isinstance(owner,Native):
            reply=owner.quality(value)
            if value is not None:self._note_quality(value)
            return reply
        return dict(applied=False,deferred=True,reason='Preparing handoff; setting will apply before native decode',requested=self.quality_setting)

    def _note_quality(self,value):
        if isinstance(value,dict):self.provenance='phase-bias'
        elif value not in (None,0,0.0,'max_quality') and self.provenance=='unbiased':self.provenance='manual-bias'

    @contextmanager
    def prepare(self,body,*,qualification_probe=None):
        if not self.lock.acquire(blocking=False):raise RuntimeError('Another request owns this conversation')
        owner=None;baseline=None;metrics=dict(session=self.session,owner_epoch=self.epoch,releases=[])
        started=time.monotonic()
        try:
            if self.poison:raise RuntimeError('Owner epoch is poisoned: '+self.poison)
            if self.cancel.is_set():raise RuntimeError('Owner epoch was cancelled')
            rendered=render_body(body)
            baseline=self.device.gpu(0,self.cfg['gpu_uuid'])
            childcfg=dict(self.cfg,host_budget_gib=self.cfg['native_host_budget_gib'],aggregate_host_budget_gib=self.cfg['host_budget_gib'])
            restore=(self.state,self.payload) if self.state is not None else None
            owner=self.current=Native(childcfg,self.session,self.epoch,1,self.cancel.is_set,restore=restore,runner=self.device,expert_list=self.expert_list)
            metrics['native_initial_load_import_seconds']=owner.launch_seconds
            tokens=owner.render(rendered);cp=owner.checkpoint
            metrics['restored_state']=deepcopy(cp)
            def matches(checkpoint):
                return checkpoint is not None and len(tokens)>=checkpoint['position'] and prefix_digest(tokens,checkpoint['position'])==checkpoint['tokens_sha256']
            if cp is not None and not matches(cp):
                if self.anchor is None or not matches(self.anchor):raise ValueError('Rendered history changed outside the saved rewind frontier')
                cp=owner.rewind(self.journal,self.anchor);metrics['rewound_to']=cp['position']
                self.provenance=self.anchor_provenance
            base_position=cp['position'] if cp else 0;append=tokens[base_position:]
            metrics.update(prompt_tokens=len(tokens),reused_prefix_tokens=base_position,new_tokens=len(append))
            metrics['base_state']=deepcopy(cp)
            routing_tokens=len(append)
            if 'rewound_to' in metrics and rendered.get('messages') and rendered['messages'][-1].get('role')=='user':
                # Removing old reasoning replays the preceding answer suffix.
                # The owner's short-new-message rule still applies: that old
                # answer is retained history, not a newly pasted document.
                blank=deepcopy(rendered);blank['messages'][-1]['content']=''
                routing_tokens=min(len(append),changed_span_tokens(tokens,owner.render(blank)))
                metrics['routing_policy']='new user text after bounded history-policy rewind'
                metrics['replayed_suffix_tokens']=len(append)-routing_tokens
            metrics['routing_tokens']=routing_tokens
            route=self.qualification_route or ('ultra' if routing_tokens>=self.threshold else 'native')
            metrics['route']=route;metrics['crossover_tokens']=self.threshold
            if route=='ultra' and append:
                if cp is not None:metrics['native_base_export']=self._checkpoint(owner,cp)
                metrics['releases'].append(release_barrier(owner,self.device,self.cfg,baseline,self.cancel.is_set));owner=self.current=None
                base=(self.state,self.contract) if cp is not None else None
                # --- DS4COMBO fast switch: resume a kept producer instead of launching one ---
                if self.kept_producer is not None and base is not None:
                    owner=self.current=self.kept_producer;self.kept_producer=None
                    metrics['producer_resume']=owner.resume(base,append)
                    metrics['ultra_load_import_prefill_seconds']=owner.prefill_seconds
                else:
                    metrics['producer_host_admission']=wait_for_host_capacity(
                        int(self.cfg.get('producer_host_budget_gib',8)*(1<<30)),int(self.cfg['min_available_gib']*(1<<30)),self.cancel.is_set)
                    owner=self.current=ComboProducer(self.cfg,self.session,self.epoch,append,self.cancel.is_set,base=base,runner=self.device)
                    metrics['ultra_load_import_prefill_seconds']=owner.prefill_seconds+metrics['producer_host_admission']['seconds']
                metrics['ultra_directory']=str(owner.directory)
                metrics['ultra_export']=self._checkpoint(owner,owner.checkpoint)
                if self.cfg.get('producer_keep_alive'):
                    metrics['producer_suspend']=owner.suspend()
                    metrics['releases'].append(suspend_barrier(owner,self.device,self.cfg,baseline,self.cancel.is_set))
                    self.kept_producer=owner;owner=self.current=None
                else:
                    metrics['releases'].append(release_barrier(owner,self.device,self.cfg,baseline,self.cancel.is_set));owner=self.current=None
                # --- end DS4COMBO fast switch ---
                owner=self.current=Native(childcfg,self.session,self.epoch,1,self.cancel.is_set,restore=(self.state,self.payload),runner=self.device,expert_list=self.expert_list)
                metrics['native_load_import_seconds']=owner.launch_seconds
            else:
                sync_started=time.monotonic()
                for off in range(0,len(append),49152):owner.sync(append[off:off+49152])
                metrics['native_prefill_seconds']=time.monotonic()-sync_started
                metrics['native_prefill_tok_s']=len(append)/metrics['native_prefill_seconds'] if append else None
            if self.quality_setting is not None:owner.quality(self.quality_setting)
            if self.journal is None:self.journal=self._ram(24<<20)
            self.anchor=owner.snapshot();self.anchor_provenance=self.provenance;metrics['anchor']=owner.anchor(self.journal)
            metrics['decode_state']=deepcopy(self.anchor)
            if qualification_probe is not None:
                # The existing full-vector API advances the actual native
                # session. Preserve this frontier, run the qualification,
                # then reopen/restore it before ordinary HTTP decoding.
                metrics['probe_checkpoint']=self._checkpoint(owner,self.anchor)
                metrics['probe']=qualification_probe(owner)
                metrics['releases'].append(release_barrier(owner,self.device,self.cfg,baseline,self.cancel.is_set));owner=self.current=None
                owner=self.current=Native(childcfg,self.session,self.epoch,1,self.cancel.is_set,restore=(self.state,self.payload),runner=self.device,expert_list=self.expert_list)
                if self.quality_setting is not None:owner.quality(self.quality_setting)
            metrics['native_directory']=str(owner.directory)
            metrics['prepared_seconds']=time.monotonic()-started
            decoder=owner
            yield owner,rendered,metrics
            # Existing native HTTP inference owns sampling/tools/stop handling.
            # Save its actual post-decode frontier, including generated tokens.
            self._note_quality(self.quality_setting)
            self._note_quality(rendered.get('quality_speed'))
            cp=owner.snapshot();metrics['final_state']=self._checkpoint(owner,cp)
            metrics['releases'].append(release_barrier(owner,self.device,self.cfg,baseline,self.cancel.is_set));owner=self.current=None
            metrics['expert_cache_restore']=self.expert_list is not None
            metrics.update(decoder.expert_restore())
            metrics['complete_seconds']=time.monotonic()-started
        except BaseException as exc:
            self.poison=repr(exc);metrics['failure']=self.poison
            if hasattr(exc,'admission_receipt'):metrics['producer_host_admission']=exc.admission_receipt
            if self.kept_producer is not None:   # DS4COMBO fast switch: never keep a producer past a failure
                try:metrics['kept_producer_release']=self.kept_producer.close()
                except BaseException as cleanup:metrics['kept_producer_release_error']=repr(cleanup);self.failed_owner=self.kept_producer
                self.kept_producer=None
            if hasattr(exc,'owned_cleanup') and not certified(exc.owned_cleanup):
                self.failed_owner=getattr(exc,'owned_runtime',getattr(exc,'owned_process',None))
                self.poison='Unconfirmed GPU release: '+self.poison
                metrics['failure']=self.poison;metrics['failed_construction_cleanup']=exc.owned_cleanup
            raise
        finally:
            if owner is not None:
                try:metrics['failure_release']=release_barrier(owner,self.device,self.cfg,baseline,lambda:False)
                except BaseException as exc:
                    self.poison='Unconfirmed GPU release: '+repr(exc);self.failed_owner=owner
                    metrics['failure_release_error']=repr(exc)
            self.current=None;self.lock.release()

    def close(self):
        if self.lock.locked():raise RuntimeError('Cannot close conversation RAM while a request owns it')
        if self.poison and self.poison.startswith('Unconfirmed GPU release'):
            raise RuntimeError('Unconfirmed GPU owner: retain checkpoint RAM until the independent Job closes')
        if self.current is not None:raise RuntimeError('Cannot close a conversation while its GPU owner is active')
        if self.state is not None:self.state.close();self.state=None
        if self.kept_producer is not None:   # DS4COMBO fast switch
            proof=self.kept_producer.close();self.kept_producer=None
            if not certified(proof):raise RuntimeError('Kept ULTRA producer retirement unconfirmed')
        if self.journal is not None:self.journal.close();self.journal=None
