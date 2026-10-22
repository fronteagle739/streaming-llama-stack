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
from .combo_native import Native
from .combo_producer import ComboProducer
from .combo_ram import LockedRam
from .combo_state import contract_from_checkpoint,verify_payload,payload_bytes
from .combo_release import release_barrier
from .combo_crossover import qualified_threshold
from .native_broker import vendor


def prefix_digest(tokens,position):
    digest=hashlib.sha256()
    for off in range(0,position,8192):
        part=tokens[off:min(position,off+8192)];digest.update(struct.pack('<'+'I'*len(part),*part))
    return digest.hexdigest()


class ComboSession:
    def __init__(self,cfg,*,device=None,qualification_route=None):
        vendor()
        from studio.dsv41_windows_runner import _Native
        self.cfg=deepcopy(cfg);self.device=device or _Native()
        self.session=secrets.token_hex(16);self.epoch=secrets.randbits(63) or 1
        self.generation=0;self.state=None;self.contract=None;self.payload=None
        self.journal=None;self.anchor=None;self.current=None;self.poison=None
        self.cancel=threading.Event();self.lock=threading.Lock();self.quality_setting=None
        if qualification_route not in (None,'native','ultra'):raise ValueError('Unknown explicit qualification route')
        self.qualification_route=qualification_route
        self.threshold=qualified_threshold(cfg) if qualification_route is None else None

    def _ram(self,size):
        return LockedRam(size,session=self.session,host_budget_gib=self.cfg['host_budget_gib'],min_available_gib=self.cfg['min_available_gib'])

    def _checkpoint(self,owner,cp):
        self.generation+=1
        contract=contract_from_checkpoint(cp,self.cfg['combo_identity'],self.session,self.epoch,self.generation,'phase-bias' if self.quality_setting else 'unbiased')
        # Reuse an admitted larger section. Never allocate a second full state.
        needed=payload_bytes(cp['position'])
        if self.state is not None and self.state.capacity<needed:self.state.close();self.state=None
        if self.state is None:self.state=self._ram(needed)
        exported=owner.export(self.state)
        payload=verify_payload(self.state,contract,cp.get('payload_sha256'))
        self.contract,self.payload=contract,payload
        return dict(export=exported,payload=payload)

    def quality(self,value=None):
        owner=self.current
        if value is not None:self.quality_setting=deepcopy(value)
        if isinstance(owner,Native):return owner.quality(value)
        return dict(applied=False,deferred=True,reason='ULTRA owns the GPU; native setting will apply before decode',requested=self.quality_setting)

    @contextmanager
    def prepare(self,body):
        if not self.lock.acquire(blocking=False):raise RuntimeError('Another request owns this conversation')
        owner=None;baseline=None;metrics=dict(session=self.session,owner_epoch=self.epoch,releases=[])
        started=time.monotonic()
        try:
            if self.poison:raise RuntimeError('Owner epoch is poisoned: '+self.poison)
            if self.cancel.is_set():raise RuntimeError('Owner epoch was cancelled')
            baseline=self.device.gpu(0,self.cfg['gpu_uuid'])
            childcfg=dict(self.cfg,host_budget_gib=self.cfg['native_host_budget_gib'])
            restore=(self.state,self.payload) if self.state is not None else None
            owner=self.current=Native(childcfg,self.session,self.epoch,1,self.cancel.is_set,restore=restore,runner=self.device)
            rendered=deepcopy(body)
            if rendered.pop('preserve_reasoning',True) is False:
                for message in rendered.get('messages',[]):
                    if message.get('role')=='assistant':message.pop('reasoning_content',None)
            tokens=owner.render(rendered);cp=owner.checkpoint
            def matches(checkpoint):
                return checkpoint is not None and len(tokens)>=checkpoint['position'] and prefix_digest(tokens,checkpoint['position'])==checkpoint['tokens_sha256']
            if cp is not None and not matches(cp):
                if self.anchor is None or not matches(self.anchor):raise ValueError('Rendered history changed outside the saved rewind frontier')
                cp=owner.rewind(self.journal,self.anchor);metrics['rewound_to']=cp['position']
            base_position=cp['position'] if cp else 0;append=tokens[base_position:]
            metrics.update(prompt_tokens=len(tokens),reused_prefix_tokens=base_position,new_tokens=len(append))
            route=self.qualification_route or ('ultra' if len(append)>=self.threshold else 'native')
            metrics['route']=route;metrics['crossover_tokens']=self.threshold
            if route=='ultra' and append:
                if cp is not None:self._checkpoint(owner,cp)
                metrics['releases'].append(release_barrier(owner,self.device,self.cfg,baseline,self.cancel.is_set));owner=self.current=None
                base=(self.state,self.contract) if cp is not None else None
                owner=self.current=ComboProducer(self.cfg,self.session,self.epoch,append,self.cancel.is_set,base=base,runner=self.device)
                metrics['ultra_load_import_prefill_seconds']=owner.prefill_seconds
                metrics['ultra_export']=self._checkpoint(owner,owner.checkpoint)
                metrics['releases'].append(release_barrier(owner,self.device,self.cfg,baseline,self.cancel.is_set));owner=self.current=None
                owner=self.current=Native(childcfg,self.session,self.epoch,1,self.cancel.is_set,restore=(self.state,self.payload),runner=self.device)
                metrics['native_load_import_seconds']=owner.launch_seconds
            else:
                for off in range(0,len(append),49152):owner.sync(append[off:off+49152])
            if self.quality_setting is not None:owner.quality(self.quality_setting)
            if self.journal is None:self.journal=self._ram(24<<20)
            self.anchor=owner.snapshot();metrics['anchor']=owner.anchor(self.journal)
            metrics['prepared_seconds']=time.monotonic()-started
            yield owner,rendered,metrics
            # Existing native HTTP inference owns sampling/tools/stop handling.
            # Save its actual post-decode frontier, including generated tokens.
            cp=owner.snapshot();metrics['final_state']=self._checkpoint(owner,cp)
            metrics['releases'].append(release_barrier(owner,self.device,self.cfg,baseline,self.cancel.is_set));owner=self.current=None
            metrics['complete_seconds']=time.monotonic()-started
        except BaseException as exc:
            self.poison=repr(exc);metrics['failure']=self.poison
            raise
        finally:
            if owner is not None:
                try:metrics['failure_release']=release_barrier(owner,self.device,self.cfg,baseline,lambda:False)
                except BaseException as exc:self.poison='Unconfirmed GPU release: '+repr(exc)
            self.current=None;self.lock.release()

    def close(self):
        if self.current is not None:raise RuntimeError('Cannot close a conversation while its GPU owner is active')
        if self.state is not None:self.state.close();self.state=None
        if self.journal is not None:self.journal.close();self.journal=None
