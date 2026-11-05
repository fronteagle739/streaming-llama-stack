"""Immutable model-bound vector registrations for the fast single engine.

Registration stores directions, never source-quant weight products. Each selected
stack becomes an immutable launch packet; its digest binds the producer and KV.
The HTTP handler authenticates requests before calling this module.
"""
import hashlib
import math
from pathlib import Path
import re
import struct
import threading

METHOD='target-weight-rank-one-v1'
HEX=re.compile(r'[a-f0-9]{64}')


class Registry:
    def __init__(self, model, directory, *, model_info=None):
        if model_info is None:
            from .native_broker import vendor
            vendor()
            from studio.direction_identity import read
            model_info=read(model)
        self.info=dict(model_info)
        if self.info.get('layers')!=40 or self.info.get('hidden')!=5120:
            raise ValueError('Writer directions require DSV4.1 1017 geometry')
        if not HEX.fullmatch(self.info.get('native_id','')):
            raise ValueError('A complete verified model identity is required')
        self.directory=Path(directory)
        self.entries={}
        self.default_rows=[]
        self.lock=threading.RLock()

    def model(self, *, capture=False, attach=False):
        return dict(native_id=self.info['native_id'],layers=40,hidden=5120,
                    capture=capture,attach=attach,method=METHOD,
                    vocab=129280,
                    source_quant=self.info.get('source_quant',''))

    def register(self, payload):
        payload=bytes(payload)
        size=48+40*2*(5120+1)*4
        if len(payload)!=size or struct.unpack_from('<4I',payload)!=(1,1,40,5120):
            raise ValueError('One complete direction registration is required')
        if payload[16:48]!=bytes.fromhex(self.info['native_id']):
            raise ValueError('Direction belongs to a different model identity')
        signal=False
        for offset in range(48,len(payload),(5120+1)*4):
            if struct.unpack_from('<f',payload,offset)[0]!=1:
                raise ValueError('Registration strengths must be unscaled')
            norm=0.0
            for (value,) in struct.iter_unpack('<f',payload[offset+4:offset+(5120+1)*4]):
                if not math.isfinite(value):raise ValueError('Nonfinite direction vector')
                norm+=value*value
            if norm>1e-12 and abs(norm-1)>1e-3:
                raise ValueError('Direction vectors must be unit length or zero')
            signal=signal or norm>1e-12
        if not signal:raise ValueError('Direction has no signal')
        identifier=hashlib.sha256(payload).hexdigest()
        with self.lock:
            if identifier not in self.entries and len(self.entries)>=256:
                raise ValueError('Direction bank is full; restart before registering more')
            self.entries.setdefault(identifier,payload)
        return dict(registered=True,id=identifier)

    def select(self, rows):
        if not isinstance(rows,list) or len(rows)>8:
            raise ValueError('Choose at most eight registered directions')
        selected=[]
        seen=set()
        for row in rows:
            if not isinstance(row,dict) or set(row)!={'id','strength'} or not isinstance(row['id'],str) or not HEX.fullmatch(row['id']):
                raise ValueError('Direction selection needs an ID and strength')
            value=row['strength']
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or not 0<=value<=4:
                raise ValueError('Direction strength must be finite from 0 to 4')
            if row['id'] in seen:raise ValueError('Duplicate direction ID')
            seen.add(row['id'])
            # Bind the exact FP32 strength that the GPU will consume.
            value=struct.unpack('<f',struct.pack('<f',value))[0]
            if value>0:selected.append((row['id'],value))
        selected.sort()
        if not selected:return dict(key='',path=None,count=0,selection=[])
        with self.lock:
            payloads=[]
            for identifier,value in selected:
                if identifier not in self.entries:
                    raise ValueError('Direction ID is not registered; re-register it on this engine')
                payloads.append(self.entries[identifier])
        header=struct.pack('<4I',1,len(selected),40,5120)+bytes.fromhex(self.info['native_id'])
        chunks=[header]
        for site in range(80):
            for (_,strength),payload in zip(selected,payloads):
                offset=48+site*(5120+1)*4
                chunks.extend((struct.pack('<f',strength),payload[offset+4:offset+(5120+1)*4]))
        packet=b''.join(chunks)
        key=hashlib.sha256(packet).hexdigest()
        self.directory.mkdir(parents=True,exist_ok=True)
        target=self.directory/(key+'.bank')
        try:
            with target.open('xb') as stream:
                stream.write(packet);stream.flush()
                import os
                os.fsync(stream.fileno())
        except FileExistsError:
            with target.open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=key:
                    raise ValueError('Immutable direction launch packet failed integrity validation')
        return dict(key=key,path=str(target.resolve()),count=len(selected),
                    selection=[dict(id=identifier,strength=value)for identifier,value in selected])

    def set_stack(self,payload):
        """Legacy explicit stack control, represented by immutable registrations."""
        payload=bytes(payload)
        if len(payload)<48:raise ValueError('Truncated direction stack')
        version,count,layers,width=struct.unpack_from('<4I',payload)
        if (version,layers,width)!=(1,40,5120) or count>8 or payload[16:48]!=bytes.fromhex(self.info['native_id']):
            raise ValueError('Direction stack geometry/model identity differs')
        stride=(width+1)*4
        if len(payload)!=48+80*count*stride:raise ValueError('Direction stack extent differs')
        rows=[]
        for adapter in range(count):
            chunks=[struct.pack('<4I',1,1,40,5120)+payload[16:48]];strength=None
            for site in range(80):
                offset=48+(site*count+adapter)*stride
                value=struct.unpack_from('<f',payload,offset)[0]
                if not math.isfinite(value) or not 0<=value<=4 or (strength is not None and value!=strength):
                    raise ValueError('Legacy stack needs one finite strength per adapter')
                strength=value;chunks.append(struct.pack('<f',1)+payload[offset+4:offset+stride])
            registered=self.register(b''.join(chunks));rows.append(dict(id=registered['id'],strength=strength))
        selected=self.select(rows)
        with self.lock:self.default_rows=rows
        return dict(applied=True,**selected)


def capture_plan(payload,*,vocab=129280):
    """Decode Studio's bounded LE32 raw/mean capture packets without reinterpretation."""
    if not isinstance(payload,(bytes,bytearray)) or len(payload)%4 or len(payload)>32784 or len(payload)<12:
        raise ValueError('Malformed direction capture packet')
    values=struct.unpack('<'+'I'*(len(payload)//4),payload)
    version,count,selected=values[:3]
    if version not in (1,2) or not 1<=count<=2048 or not 1<=selected<=2048:
        raise ValueError('Capture needs1..2048 tokens and selected positions')
    start=4 if version==2 else 3
    group_count=values[3] if version==2 else selected
    if not 1<=group_count<=16:raise ValueError('Capture needs1..16 groups')
    expected=start+count+selected*(2 if version==2 else 1)
    if len(values)!=expected:raise ValueError('Capture packet extent differs')
    tokens=list(values[start:start+count]);tail=values[start+count:]
    if any(token>=vocab for token in tokens):raise ValueError('Capture token is outside the native vocabulary')
    positions=list(tail[::2]) if version==2 else list(tail)
    groups=list(tail[1::2]) if version==2 else list(range(selected))
    if positions!=sorted(set(positions)) or positions[-1]>=count or set(groups)!=set(range(group_count)):
        raise ValueError('Capture positions/groups must be sorted, unique, in range and complete')
    raw=struct.pack('<'+'I'*count,*tokens)
    plan=dict(schema='writer-capture-v1',tokens_sha256=hashlib.sha256(raw).hexdigest(),tokens_count=count,
              positions=positions,groups=groups,group_count=group_count,means=version==2)
    return tokens,plan


def evaluate_plan(payload):
    import json
    if len(payload)<4:raise ValueError('Truncated direction evaluation packet')
    count=struct.unpack_from('<I',payload)[0]
    if not 1<=count<=8192 or len(payload)<4+count+12:raise ValueError('Evaluation selection exceeds its bound')
    selection=json.loads(payload[4:4+count])
    if not isinstance(selection,dict) or set(selection)!={'direction_adapters'}:
        raise ValueError('Evaluation requires an immutable direction selection')
    tokens,plan=capture_plan(payload[4+count:])
    if plan['means']:raise ValueError('Logit evaluation requires raw positions')
    return tokens,plan,selection['direction_adapters']


def capture_headers(plan,*,evaluate=False):
    floats=40*6*5120+(5120+129280 if evaluate else 0)
    return {'Content-Type':'application/octet-stream','X-DS4-Feature-Kind':'writer-hc-logits-v1' if evaluate else 'writer-hc-v1',
        'X-DS4-Layers':'40','X-DS4-Hidden':'5120','X-DS4-HC-Streams':'4',
        'X-DS4-Floats-Per-Position':str(floats),'X-DS4-Selected-Count':str(len(plan['positions'])),
        'X-DS4-Group-Count':str(plan['group_count']),'X-DS4-Aggregation':'mean' if plan['means'] else 'raw'}


def bank_arguments(cfg):
    """Fail before launch if a packet or its immutable digest was changed."""
    key=cfg.get('direction_bank_sha256','');filename=cfg.get('direction_bank_file')
    if not key and not filename:return {}
    if not isinstance(key,str) or not HEX.fullmatch(key) or not filename:raise ValueError('Direction bank needs its full digest and packet')
    packet=Path(filename)
    if not 48<packet.stat().st_size<=48+80*8*5121*4:raise ValueError('Direction launch packet exceeds the bounded bank format')
    with packet.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=key:raise ValueError('Direction launch packet digest differs')
    return {'--direction-bank-file':str(packet.resolve()),'--direction-bank-sha256':key}


def verify_ready(cfg,ready):
    key=cfg.get('direction_bank_sha256','')
    qualified=cfg.get('writer_directions_qualified_sha256')
    required=bool(key) or bool(qualified and qualified==cfg.get('producer_sha256'))
    if required and (not isinstance(ready,dict) or ready.get('direction_binding')!=key):
        raise ValueError('Producer did not acknowledge the exact immutable direction bank')


def token_offsets(tokenizer,text):
    """Compute exact UTF-8 offsets from the native tokenizer's emitted pieces."""
    if not isinstance(text,str):raise ValueError('Text must be a string')
    raw=text.encode('utf-8','strict')
    ids=tokenizer.text(text)
    pieces=[tokenizer.token_bytes(identifier)for identifier in ids]
    if b''.join(pieces)!=raw:
        raise ValueError('Tokenizer pieces do not reproduce the input text exactly')
    position=0;offsets=[]
    for piece in pieces:
        offsets.append([position,position+len(piece)]);position+=len(piece)
    return dict(tokens=ids,byte_offsets=offsets,text_bytes=len(raw))


def qualified(cfg):
    """A deployment must explicitly bind qualification to its actual producer."""
    return bool(cfg.get('producer_sha256') and
                cfg.get('writer_directions_qualified_sha256') == cfg['producer_sha256'])


def device_reserve(count):
    """Permanent target factors plus both banks, including the external decoder.

    Prefill and native decode each retain the full target-factor closure.
    Native decode also retains the dense writer working set. Charge
    future growth separately from allocations already visible in physical free.
    """
    if type(count) is not int or not 0 <= count <= 8:
        raise ValueError('Direction bank count must be0..8')
    if not count:return 0
    native_floats=40*(8192+2304+384*2304)+80*5121
    prefill_floats=40*(8192+2304+384*2304)+80*5121
    dense_bytes=40*5120*(8192+2304)*4
    return count*(native_floats+prefill_floats)*4+dense_bytes+(8<<20)


def prefill_device_reserve(count):
    """Prefill's factor closure; decode cache is allocated after compaction."""
    if type(count) is not int or not 0 <= count <= 8:
        raise ValueError('Direction bank count must be0..8')
    if not count:return 0
    factor_floats=40*(8192+2304+384*2304)+80*5121
    return count*factor_floats*4+(8<<20)


def apply_selection(engine, rows):
    """Retire/replay on a bank change; a KV frontier never changes its bank."""
    selected = engine.directions().select(rows)
    if selected['key'] != engine.cfg.get('direction_bank_sha256', ''):
        if engine.producer is not None:
            engine.save_snapshot('conversation')
            engine.retire('writer direction bank changed')
        engine.pending_snapshot = None
        engine.tool_memory = {}
        engine.cut_turn = None
        engine.cfg.update(direction_bank_sha256=selected['key'],
                          direction_bank_file=selected['path'], direction_bank_count=selected['count'])
    return selected


def capture(engine, payload, *, evaluate=False):
    """Capture through the normal streaming producer and its ordinary guardian.

    Capture adds graph consumers; its fetch provider, kernels and carry remain
    those of normal serving. Dedicated producers prevent a capture changing a
    conversation's live or persisted direction selection.
    """
    from copy import deepcopy
    import secrets
    from .journal import atomic_json
    registry = engine.directions()
    if evaluate:
        tokens, plan, rows = evaluate_plan(payload)
    else:
        tokens, plan = capture_plan(payload)
        rows = list(registry.default_rows)
    selected = registry.select(rows)
    if engine.producer is not None:
        engine.save_snapshot('conversation')
        engine.retire('dedicated writer capture')
    directory = Path(engine.cfg['runtime_root']) / ('writer-capture-' + secrets.token_hex(8))
    directory.mkdir()
    atomic_json(directory / 'plan.json', plan)
    cfg = deepcopy(engine.cfg)
    cfg.update(direction_bank_sha256=selected['key'], direction_bank_file=selected['path'],
               direction_bank_count=selected['count'],
               writer_capture_plan=str(directory / 'plan.json'),
               writer_capture_output=str(directory / 'capture.f32'))
    if evaluate: cfg['writer_evaluate_output'] = str(directory / 'evaluate.f32')
    original_cfg = engine.cfg
    try:
        engine.cfg = cfg
        engine.launch(tokens)
    finally:
        engine.cfg = original_cfg
    try:
        headers = capture_headers(plan, evaluate=evaluate)
        size = int(headers['X-DS4-Floats-Per-Position']) * plan['group_count'] * 4
        if size > 128 * 1024**2: raise ValueError('Capture exceeds the 128 MiB response bound')
        target = directory / ('evaluate.f32' if evaluate else 'capture.f32')
        if target.stat().st_size != size: raise ValueError('Native capture extent differs')
        data = target.read_bytes()
        if any(not math.isfinite(x[0]) for x in struct.iter_unpack('<f', data)):
            raise ValueError('Native capture returned nonfinite features')
        return data, headers
    finally:
        # Launch/retire preserve the owned hold on any uncertified cleanup.
        engine.retire('dedicated writer capture finished')
